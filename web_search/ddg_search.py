# ==========================================================================================================
# Web Search Module (DuckDuckGo)
# ==========================================================================================================
# 용도: 로컬 검색 실패 시 웹 검색 수행
# 노트북 복원: DuckDuckGoSearchRun() 사용
# ==========================================================================================================

from langchain_community.tools import DuckDuckGoSearchRun

# ==========================================
# DuckDuckGo 검색 도구
# ==========================================
web_search_tool = DuckDuckGoSearchRun()

def run_web_search(query):
    """
    DuckDuckGo 검색 실행 (노트북과 동일)
    
    Args:
        query (str): 검색 쿼리
    
    Returns:
        List[Dict]: [{"content": str, "url": str, "title": str}]
    """
    print(f"   [Search] '{query}' 검색 중 (DuckDuckGo)...")
    try:
        content = web_search_tool.invoke(query)
        return [{
            "content": content,
            "url": "https://duckduckgo.com",
            "title": "DuckDuckGo Web Search"
        }]
    except Exception as e:
        print(f"   [Error] 검색 실패: {e}")
        return []


if __name__ == "__main__":
    # 테스트
    results = run_web_search("나라장터 학사정보시스템")
    
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r['title']}")
        print(f"URL: {r['url']}")
        print(f"내용: {r['content'][:100]}...")
