from langgraph.graph import END, StateGraph
from agent.state import GraphState
from agent.nodes import (
    retrieve_local,
    grade_documents,
    rewrite_query,
    retrieve_local_retry,
    web_search_node,
    generate
)

# ==========================================\n
# 엣지 조건 (Conditional Edge)
# ==========================================\n
def decide_after_grade(state) -> str:
    """
    Grade 결과에 따라 다음 단계 결정:
    - Yes → Generate (바로 답변 생성)
    - No + retry_count < 1 → Rewrite (로컬 재검색)
    - No + retry_count >= 1 → Web Search (웹으로 이동)
    """
    relevance = state.get("relevance")
    retry_count = state.get("retry_count", 0)
    
    if relevance == "yes":
        print("   -> [Route] 적합 → 생성")
        return "generate"
    elif retry_count < 1:
        print("   -> [Route] 부적합 → 로컬 재검색")
        return "rewrite_query"
    else:
        print("   -> [Route] 부적합 (2차 실패) → 웹 검색")
        return "web_search_node"

# ==========================================\n
# 그래프 빌드
# ==========================================\n
workflow = StateGraph(GraphState)

# 1. 노드 추가
workflow.add_node("retrieve_local", retrieve_local)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_local_retry", retrieve_local_retry)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate", generate)

# 2. 엣지 연결
workflow.set_entry_point("retrieve_local")
workflow.add_edge("retrieve_local", "grade_documents")

# 3. 조건부 엣지 (Grade 후 분기)
workflow.add_conditional_edges(
    "grade_documents",
    decide_after_grade,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
        "web_search_node": "web_search_node"
    }
)

# 4. 나머지 엣지
workflow.add_edge("rewrite_query", "retrieve_local_retry")
workflow.add_edge("retrieve_local_retry", "grade_documents")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# 5. 컴파일
app = workflow.compile()
print(">>> [Graph] LangGraph 빌드 완료.")
