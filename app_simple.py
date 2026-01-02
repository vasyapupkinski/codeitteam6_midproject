
# ==========================================================================================================
# Simple RAG Pipeline (Jupyter Notebook Style)
# ==========================================================================================================
# 특징: LangGraph 제거, 복잡한 에이전트 로직 제거, 즉시 스트리밍, 노트북과 동일한 품질
# ==========================================================================================================

import sys
import os
import gradio as gr
from threading import Thread

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 기존 모듈 재사용
from config.settings import GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_SHARE, CHUNK_PREVIEW_LENGTH
from data.loader import load_documents
from embedding.bge_embedder import BGEEmbedder
from vectorstore.loader import load_qdrant_db
from retrieval.dense_retriever import create_dense_retriever
from retrieval.sparse_retriever import create_sparse_retriever
from retrieval.hybrid_retriever import create_hybrid_retriever
from filtering.date_filter import filter_by_date
from filtering.budget_filter import filter_by_budget
from filtering.contract_filter import filter_private_contracts
from reranking.reranker import Reranker
from utils.formatters import format_docs_for_llm
from utils.diversity import get_diverse_top_k
from llm.gemma import GemmaLLM
from llm.qwen import QwenLLM
from web_search.ddg_search import run_web_search

# ==========================================================================================================
# [전역 초기화] (서버 시작 시 1회 실행)
# ==========================================================================================================
print("\n" + "="*80)
print(" Simple RAG Pipeline 초기화 중... (Notebook Mode)")
print("="*80)

documents = load_documents()
embedder = BGEEmbedder()
vector_store = load_qdrant_db(embedding_function=embedder.get_embedding_function())

dense_retriever = create_dense_retriever(vector_store)
sparse_retriever = create_sparse_retriever(documents)
hybrid_retriever = create_hybrid_retriever(dense_retriever, sparse_retriever)
reranker = Reranker()

gemma_llm = GemmaLLM()
qwen_llm = QwenLLM()

llms = {
    "Gemma 2 9B": gemma_llm,
    "Qwen 2.5 7B": qwen_llm
}

print("\n" + "="*80)
print(" ✓ 초기화 완료! (스트리밍 준비됨)")
print("="*80 + "\n")


# ==========================================================================================================
# [검색 로직]
# ==========================================================================================================
def search_pipeline(query):
    # 1. 하이브리드 검색
    docs = hybrid_retriever.invoke(query)
    
    # 2. 필터링
    docs = filter_by_date(docs, query)
    docs = filter_by_budget(docs, query)
    docs = filter_private_contracts(docs, query)
    
    # 3. 문서 없으면 웹 검색 시도 (Fallback)
    if not docs:
        print("   ⚠️ 로컬 문서 없음 -> 웹 검색 시도")
        web_results = run_web_search(f"나라장터 공고 {query}", max_results=3)
        web_docs = []
        for res in web_results:
             # 웹 검색 결과를 Document 형식으로 변환
             web_docs.append({
                 "title": "웹 검색 결과",
                 "agency": "인터넷",
                 "source": "Web",
                 "chunk_id": "WEB",
                 "page_no": f"URL: {res.get('url')}",
                 "price": "웹 참조",
                 "pub_date": "실시간",
                 "end_date": "-",
                 "chunk_text": res.get('content')
             })
        # 웹 검색 결과는 Reranking 없이 바로 반환 (Simple)
        # format_docs_for_llm은 객체나 튜플을 기대하므로 형식 맞춤
        return web_docs, "web"

    # 4. Reranking & Diversity
    scored_docs = reranker.rerank(query, docs)
    final_docs = get_diverse_top_k(scored_docs)
    
    return final_docs, "local"


# ==========================================================================================================
# [생성 로직] (스트리밍)
# ==========================================================================================================
def generate_response(query, model_name):
    print(f"\n[질문]: {query}")
    
    # 1. 검색
    documents, source = search_pipeline(query)
    
    if not documents:
        yield "죄송합니다. 관련 정보를 찾을 수 없습니다."
        return

    # 2. 컨텍스트 구성
    # settings.CHUNK_PREVIEW_LENGTH (2000자) 적용됨
    context = format_docs_for_llm(documents)
    
    # 3. 프롬프트 구성 (Jupyter와 동일)
    system_prompt = """
    # 페르소나
    당신은 대한민국 공공기관 입찰 및 RFP(제안요청서) 분석 분야에서 20년 경력을 가진 '수석 컨설턴트'입니다.
    방대한 문서에서 핵심 정보를 정확히 추출하고, 입찰 성공을 위한 전략적 통찰을 제공하는 전문가입니다.
    # 임무
    제공된 [검색된 문서 정보]만을 바탕으로 사용자의 질문에 답하세요. 
    당신의 분석은 기업의 입찰 전략 수립에 직접적으로 활용되므로, 정보의 정확성과 전략적 깊이가 동시에 요구됩니다.
    # 지시사항 및 제약조건 (필수 준수)
    1. **검색 범위 및 출처 명시 (필수)**:
       - 답변 시작 시 반드시 "총 X개 문서를 검색했으며, 그 중 Y개를 분석했습니다"라고 명시하십시오.
       - 모든 정보의 끝에는 반드시 [청크 ID]와 [사업명]을 병기하여 검증 가능하게 하십시오. 
         (예: "본 사업은 ... (출처: proj_001_chk_0023, 사업명: 00 구축사업)")
    2. **데이터 신뢰도 우선순위**:
       - 문서 상단에 **★표시가 된 메타데이터(예산, 날짜, 기관, 공고번호)**는 검증된 확정 정보입니다. 
       - 본문 내용과 메타데이터가 충돌할 경우, 반드시 **메타데이터를 최우선으로 신뢰**하여 답변하십시오.
       - **중요**: 메타데이터는 절대 바꿔 말하지 말고 원문 그대로 사용하십시오. (특히 예산이 "수의계약"으로 되어있으면 그대로 언급)
    3. **전략적 분석 수행 (컨설팅)**:
       - 분석된 각 사업에 대해 수석 컨설턴트의 시각에서 **'입찰 전략'**과 수행 시 예상되는 **'위험 요소(Risk)'**를 반드시 구체적으로 언급하십시오.
    4. **질의서(Q&A) 항목 제안**:
       - 입찰 준비 및 현장 설명회 과정에서 발주기관에 확인이 필요한 **'실무 질의서 항목'**을 각 사업별로 최소 2개 이상 제안하십시오.
    5. **노이즈 필터링**:
       - 실질적인 사업 요구사항이 없는 단순 서식, 별지 양식, 신청서 예시 등은 분석 대상에서 과감히 제외하십시오.
    6. **정직성 및 할루시네이션 방지**:
       - 제공된 문서 내에 답변을 위한 근거가 없거나 불충분한 경우, 지어내지 말고 "제공된 문서에서는 해당 정보를 찾을 수 없습니다"라고 솔직하게 답변하십시오.
    7. **답변 구조화**:
       - 입찰 건 목록 제시 시: [사업명 / 발주기관 / 예산 / 마감일 / 출처] 형식을 유지하십시오.
       - 핵심 요구사항 및 전략/리스크 비교 시 가독성을 위해 **표(Table)** 형식을 적극 활용하십시오.
    # 사고 과정 (Step-by-Step)
    1) 질문의 핵심 키워드 및 사용자 의도 파악
    2) 검색된 문서 중 유효한 문서(서식 제외) 선별 및 개수 카운트
    3) ★메타데이터 기반으로 사업별 기본 팩트(예산, 일정, 공고번호 등) 확정
    4) 본문 내 세부 요구사항 추출 및 사업 수행의 전략/리스크 요인 분석
    5) 사업별 실무 질의 사항(Q&A) 도출
    6) 출처를 포함하여 수석 컨설턴트 톤으로 최종 답변 작성
    """

    user_prompt = f"""
[검색된 문서]
{context}

[질문]
{query}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 4. 스트리밍 생성
    llm = llms[model_name]
    for chunk in llm.stream(messages):
        yield chunk


# ==========================================================================================================
# [Gradio UI]
# ==========================================================================================================
with gr.Blocks(title="Simple RAG Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## ⚡ Fast & Simple RAG (Jupyter Notebook Mode)")
    
    with gr.Row():
        query_input = gr.Textbox(label="질문", placeholder="질문을 입력하세요...", lines=3)
        model_dropdown = gr.Dropdown(choices=["Gemma 2 9B", "Qwen 2.5 7B"], value="Gemma 2 9B", label="모델")
    
    submit_btn = gr.Button("🚀 실행 (스트리밍)", variant="primary")
    output = gr.Textbox(label="답변", lines=25)

    submit_btn.click(
        fn=generate_response,
        inputs=[query_input, model_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=GRADIO_SHARE
    )
