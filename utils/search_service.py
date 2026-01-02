# ==========================================================================================================
# Search Service Module
# ==========================================================================================================
# 용도: 하이브리드 검색 + 필터링 + Reranking + Diversity 파이프라인
# 입력: Query (str)
# 출력: List[(Document, score)] - 최종 상위 K개 문서
# ==========================================================================================================

from dependencies import hybrid_retriever, reranker
from filtering.date_filter import filter_by_date
from filtering.budget_filter import filter_by_budget
from filtering.contract_filter import filter_private_contracts
from utils.diversity import get_diverse_top_k
from config.settings import TOP_K_FINAL


def search_and_rerank(query, debug=True):
    """
    [메인 검색 함수] Hybrid -> Filtering -> Reranking -> Diversity
    """
    # ... (생략) ...
    
    # ========================================
    # [1] Hybrid Retrieval
    # ========================================
    docs = hybrid_retriever.invoke(query)
    if debug:
        print(f"   1. 1차 검색: {len(docs)}개")
    
    # ========================================
    # [2] Hard Filtering
    # ========================================
    docs = filter_by_date(docs, query)
    docs = filter_by_budget(docs, query)
    docs = filter_private_contracts(docs, query)
    
    if not docs:
        if debug:
            print("   [Result] 검색 결과 없음")
        return []
    
    # ========================================
    # [3] Reranking
    # ========================================
    # Reranker 클래스의 rerank 메서드 사용 (내부적으로 sorted list 반환)
    scored_docs = reranker.rerank(query, docs)
    
    # ========================================
    # [4] Diversity (Project Deduplication)
    # ========================================
    final_docs = get_diverse_top_k(scored_docs, k=TOP_K_FINAL)
    
    # ========================================
    # [5] Debug Output
    # ========================================
    if debug:
        print(f"   3. 최종 결과: Top {len(final_docs)}\n")
        print(f" {'[순위]':<6} {'[점수]':<8} {'[금액]':<25} {'[제목]':<30}")
        print("-" * 85)
        for i, (doc, score) in enumerate(final_docs):
            title = doc.metadata.get('title', '')[:25]
            price = str(doc.metadata.get('price', '미정'))[:22]
            print(f" {i+1:<6} {score:.4f}   {price:<25} {title:<30}")
            print("-" * 85)
    
    return final_docs
