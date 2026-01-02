from typing import List, TypedDict, Literal

class GraphState(TypedDict):
    """
    LangGraph 상태 정의
    """
    question: str               # 원본 질문
    rewritten_question: str     # 재작성된 질문 (2차 검색용)
    generation: str             # 최종 답변
    documents: List[object]     # 문서 리스트
    search_source: str          # 출처 (local/local_retry/web)
    relevance: str              # 문서 적합성 (yes/no)
    retry_count: int            # 재시도 횟수 (최대 1회)
    model_name: str             # 사용할 LLM 모델 이름
