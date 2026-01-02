# ==========================================================================================================
# RAG Application - GCP Production Ready (팀원 검증 완료)
# ==========================================================================================================
import sys
import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 팀원 방식: config.py에서 설정만, main.py에서 초기화
from config.settings import GRADIO_SERVER_NAME, GRADIO_SERVER_PORT
# [중요] dependencies에서 이미 초기화된 객체들을 가져옴 (Double Init 방지)
from dependencies import documents, embedder, vector_store, hybrid_retriever, reranker
from agent.graph import app as agent_app
from config.settings import EMBEDDING_MODEL

print("\n" + "="*80)
print(" RAG Agent (Gradio) 시작")
print("="*80 + "\n")

# dependencies.py가 임포트되면서 이미 초기화 로그가 출력되었습니다.
print(f">>> 현재 설정: {EMBEDDING_MODEL}")
print(f">>> 문서 수: {len(documents)}")


print(">>> 초기화 완료!\n")

# Gradio 인터페이스
# Gradio 인터페이스 (스트리밍 적용)
async def chat_interface(message, history, model_name):
    """
    LangGraph 기반 스트리밍 채팅 인터페이스
    """
    print(f"\n[Query] {message}")
    print(f"[Model] {model_name}\n")
    
    inputs = {
        "question": message,
        "model_name": model_name,
        "retry_count": 0
    }
    
    # LangGraph v2 astream_events API 사용
    # 'on_chat_model_stream' 이벤트만 필터링하여 토큰 단위 스트리밍
    partial_message = ""
    async for event in agent_app.astream_events(inputs, version="v1"):
        kind = event["event"]
        
        # LLM이 토큰을 생성할 때마다
        if kind == "on_chat_model_stream":
            # [중요] 'generate' 노드에서 발생하는 스트리밍만 UI에 표시
            # (rewrite_query, grade_documents 등의 내부 사고 과정은 숨김)
            node_name = event.get("metadata", {}).get("langgraph_node", "")
            if node_name == "generate":
                content = event["data"]["chunk"].content
                if content:
                    partial_message += content
                    yield partial_message

# Gradio UI
with gr.Blocks(title="RAG Agent") as demo:
    gr.Markdown("# RAG Agent System\n입찰 공고 분석 에이전트")
    
    model_dropdown = gr.Dropdown(
        choices=["Gemma 2 9B", "Qwen 2.5 7B", "GPT-5-mini"],
        value="GPT-5-mini",
        label="LLM 모델"
    )
    
    # 커스텀 챗봇 컴포넌트 (높이 조절용)
    chatbot = gr.Chatbot(height=900)
    
    gr.ChatInterface(
        fn=chat_interface,
        chatbot=chatbot,
        additional_inputs=[model_dropdown],
        examples=[
            ["2024년 학사정보시스템 구축 사업", "GPT-5-mini"], 
            ["1억 이하 수의계약 사업", "GPT-5-mini"]
        ]
    )

if __name__ == "__main__":
    print("="*80)
    print(f"Gradio 서버: {GRADIO_SERVER_NAME}:{GRADIO_SERVER_PORT}")
    print("="*80 + "\n")
    
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=False
    )
