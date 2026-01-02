# ==========================================================================================================
# Agent Nodes (LangGraph)
# ==========================================================================================================
# 용도: 노트북의 6개 노드 함수를 그대로 복원
# ==========================================================================================================

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from dependencies import llms
from utils.search_service import search_and_rerank
from utils.formatters import format_docs_for_llm
from web_search.ddg_search import run_web_search
from agent.tools import get_current_date, calculate_days_until, date_tools

# ==========================================
# 노드 정의
# ==========================================

def retrieve_local(state):
    """[노드 1] 로컬 검색 (1차)"""
    print(f"\n--- [1. Local Search] 1차 검색 ---")
    question = state["question"]
    
    # Jupyter의 search_and_rerank 함수 재사용
    docs = search_and_rerank(question, debug=False)
    
    return {
        "documents": docs, 
        "question": question,
        "rewritten_question": question,
        "search_source": "local",
        "retry_count": 0
    }


def grade_documents(state):
    """[노드 2] 문서 적합성 평가 (LLM Judge)"""
    retry_count = state.get("retry_count", 0)
    print(f"--- [2. Grade] 문서 평가 (시도: {retry_count + 1}회) ---")
    
    documents = state["documents"]
    question = state.get("rewritten_question", state["question"])
    
    # 문서 없음 → 부적합
    if not documents:
        print("   -> 문서 없음 (0건)")
        return {"relevance": "no"}
    
    # LLM 선택
    model_name = state.get("model_name", "Gemma 2 9B")
    llm = llms[model_name]
    
    # Jupyter의 format_docs_for_llm 함수 재사용
    context = format_docs_for_llm(documents)
    
    # LLM에게 적합성 판단 요청 (Notebook Parity Version)
    prompt = f"""
    당신은 문서 적합성 평가자입니다.
    아래 문서가 질문에 답하기 충분한지 판단하세요.
    [질문]: {question}
    [문서 샘플]: {context[:1500]}
    판정 기준:
    - '예': 질문과 관련된 내용이 문서에 포함되어 있음 (느슨한 기준이긴 한데 자세히 보긴 할 것)
    - '아니오': 질문과 전혀 무관한 문서임(예: 단순 서식, 아예 다른 분야 공고, 논리적으로 무관한 문서나 사업, 논리적으로 다른 비용과 날짜)
    '예' 또는 '아니오'로만 답변:
    """
    
    response = llm.invoke(prompt).content.strip()
    
    if "예" in response or "Yes" in response:
        print("   -> [판정] 적합 (Yes)")
        return {"relevance": "yes"}
    else:
        print("   -> [판정] 부적합 (No)")
        return {"relevance": "no"}


def rewrite_query(state):
    """[노드 3] 질문 재작성 (로컬 재검색 전)"""
    print(f"--- [3. Rewrite] 질문 재작성 중... ---")
    original_question = state["question"]
    retry_count = state.get("retry_count", 0)
    
    # LLM 선택
    model_name = state.get("model_name", "Gemma 2 9B")
    llm = llms[model_name]
    
    # LLM에게 질문을 다르게 표현하도록 요청
    rewrite_prompt = f"""
    당신은 검색 쿼리 최적화 도구입니다. 목적은 입찰 공고 데이터베이스에서 정확한 문서를 찾기 위한 '검색어'를 만드는 것입니다.
    
    지시사항:
    1. 사용자의 복잡한 질문에서 핵심 검색 키워드(Key Entities)만 추출하세요.
    2. 불필요한 서술어, 문장 부호, 번호 매기기, 설명(사족)을 모두 제거하세요.
    3. 오직 검색 엔진에 입력할 **하나의 문자열**만 출력하세요.
    4. 가능하면 최대한 로컬 문서에서 찾게 하기(로컬 문서들에 어지간하면 있음 아예 말도 안되는거만 부적합으로 넘길 것)
    
    원본 질문: {original_question}
    
    최적화된 검색어:
    """
    
    new_question = llm.invoke(rewrite_prompt).content.strip()
    print(f"   -> 재작성: '{new_question}'")
    
    return {
        "rewritten_question": new_question,
        "retry_count": retry_count + 1
    }


def retrieve_local_retry(state):
    """[노드 4] 로컬 재검색 (2차)"""
    print(f"\n--- [4. Local Retry] 2차 검색 (재작성된 질문) ---")
    rewritten_question = state.get("rewritten_question", state["question"])
    
    # 재작성된 질문으로 다시 검색
    docs = search_and_rerank(rewritten_question, debug=False)
    
    return {
        "documents": docs,
        "search_source": "local_retry"
    }


def web_search_node(state):
    """[노드 5] 웹 검색 (최후의 수단)"""
    print(f"\n--- [5. Web Search] 로컬 2차 시도 실패 -> 웹 검색 ---")
    question = state["question"]
    web_query = f"나라장터 공고 {question}"
    
    docs = []
    web_results = run_web_search(web_query)
    
    for idx, res in enumerate(web_results):
        # Jupyter의 format_docs_for_llm 호환 메타데이터
        metadata = {
            "title": "DuckDuckGo 웹 검색",
            "agency": "인터넷",
            "source": "Web",
            "chunk_id": f"WEB-{idx+1:03d}",
            "page_no": f"URL: {res.get('url')}",
            "price": "웹 참조",
            "pub_date": "실시간",
            "end_date": "-",
            "chunk_text": res.get('content')
        }
        docs.append(Document(page_content=res.get('content'), metadata=metadata))
    
    return {"documents": docs, "search_source": "web"}


def generate(state):
    """[노드 6] 최종 답변 생성 (Stream)"""
    source = state.get("search_source", "local")
    print(f"--- [6. Generate] 답변 생성 (Source: {source}) ---")
    
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {"generation": "죄송합니다. 정보를 찾을 수 없습니다."}
    
    # LLM 선택
    model_name = state.get("model_name", "Gemma 2 9B")
    llm = llms[model_name]
    
    # Jupyter의 format_docs_for_llm 함수 재사용
    context = format_docs_for_llm(documents)
    
    # Jupyter Notebook Step 5의 완전한 System Prompt
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
    5. **노이즈 필터링 (묵묵히 수행)**:
       - 분석 대상이 아닌 문서(단순 서식, 양식 등)는 **언급하지 말고 조용히 제외**하십시오.
       - "서식이라서 제외했습니다"와 같은 **불필요한 설명(메타 코멘터리)을 절대 하지 마십시오.** 결과만 제시하십시오.
    6. **정직성 및 할루시네이션 방지**:
       - 제공된 문서 내에 답변을 위한 근거가 없거나 불충분한 경우, 지어내지 말고 "제공된 문서에서는 해당 정보를 찾을 수 없습니다"라고 솔직하게 답변하십시오.
    7. **답변 구조화**:
       - 입찰 건 목록 제시 시: [사업명 / 발주기관 / 예산 / 마감일 / 출처] 형식을 유지하십시오.
       - 핵심 요구사항 및 전략/리스크 비교 시 가독성을 위해 **표(Table)** 형식을 적극 활용하십시오.
    8. **답변 길이 및 최적화 (매우 중요)**:
       - 답변은 **최대한 약 4,000자(공백 포함)** 내외로 충실하게 작성하되, 중간에 잘리지 않도록 핵심 내용을 모두 포함하여 완결하십시오.
       - 너무 짧은 요약보다는 '상세한 컨설팅 보고서' 형태로 문맥을 풍부하게 작성하십시오.(최대한 4000자 내로 끝내보기 그리고 다른 사족은 필요없음 내용이 제일 중요함)
    # 사고 과정 (Step-by-Step)
    1) 질문의 핵심 키워드 및 사용자 의도 파악
    2) 검색된 문서 중 유효한 문서(서식 제외) 선별 및 개수 카운트
    3) ★메타데이터 기반으로 사업별 기본 팩트(예산, 일정, 공고번호 등) 확정
    4) 본문 내 세부 요구사항 추출 및 사업 수행의 전략/리스크 요인 분석
    5) 사업별 실무 질의 사항(Q&A) 도출
    6) 출처를 포함하여 수석 컨설턴트 톤으로 최종 답변 작성
    """
    
    # 소스별 추가 안내
    if source == "web":
        note = "\n\n[주의] 웹 검색 결과입니다. 원문 확인이 필요합니다."
    else:
        note = "\n\n[참고] 내부 검증된 데이터입니다."
    
    user_prompt = f"""
    [검색된 문서]
    {context}{note}
    
    [질문]
    {question}
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # 날짜 툴을 LLM에 바인딩
    llm_with_tools = llm.bind_tools(date_tools)
    
    print("\n[ 답변 생성 중...]\n")
    full_response = ""
    
    # 1차 호출: 툴 사용 여부 확인 (여기서도 streaming=True이므로 UI에 이미 출력됨)
    response = llm_with_tools.invoke(messages)
    
    # 툴 호출이 있으면 실행 후 재호출
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print("[ 날짜 계산 툴 사용 중...]\n")
        
        tool_results = []
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call.get('args', {})
            
            # 툴 실행
            if tool_name == 'get_current_date':
                result = get_current_date.invoke({})
            elif tool_name == 'calculate_days_until':
                result = calculate_days_until.invoke(tool_args)
            else:
                result = "알 수 없는 툴"
            
            tool_results.append(f"[{tool_name}] → {result}")
        
        # 툴 결과를 메시지에 추가
        tool_info = "\n".join(tool_results)
        messages.append(HumanMessage(content=f"\n\n[툴 실행 결과]\n{tool_info}"))
        
        # 2차 호출 (Stream) - 툴 결과 반영하여 최종 답변
        for chunk in llm.stream(messages):
            print(chunk.content, end="", flush=True)
            full_response += chunk.content
    else:
        # 툴 호출 없음 -> invoke 결과가 곧 답변임
        # (invoke 할 때 이미 스트리밍 이벤트가 발생했으므로 다시 stream() 하면 두 번 출력됨)
        print(response.content, end="", flush=True)
        full_response = response.content
    
    print("\n")
    return {"generation": full_response}


print(" 노드 정의 완료")
