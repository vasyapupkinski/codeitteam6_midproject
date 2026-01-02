# ==========================================================================================================
# Contract Filter Module
# ==========================================================================================================
# 용도: 수의계약 조건으로 문서 필터링
# 키워드: "수의계약", "직찰"
# ==========================================================================================================


def filter_private_contracts(docs, query):
    """
    수의계약 조건으로 문서 필터링
    
    Args:
        docs (List[Document]): 문서 리스트
        query (str): 사용자 질의
    
    Returns:
        List[Document]: 필터링된 문서 리스트
    """
    # 질의에 수의계약 관련 키워드가 있는지 확인
    if "수의계약" in query or "직찰" in query:
        filtered = [
            doc for doc in docs 
            if doc.metadata.get('is_private_contract') is True
        ]
        
        if filtered:
            print(f"   [Private Filter] 수의계약 필터링 → {len(filtered)}개")
            return filtered
    
    # 조건 없으면 원본 반환
    return docs


if __name__ == "__main__":
    # 테스트 코드
    from data.loader import load_documents
    
    docs = load_documents()
    
    # 테스트 쿼리
    test_queries = [
        "수의계약 사업",
        "직찰 건",
        "일반 입찰"  # 필터링 안 됨
    ]
    
    for q in test_queries:
        print(f"\n쿼리: {q}")
        filtered = filter_private_contracts(docs, q)
        print(f"결과: {len(filtered)}개")
        
        if filtered and filtered != docs:
            # 수의계약 예시 출력
            print("샘플:")
            for doc in filtered[:3]:
                print(f"  - {doc.metadata.get('title', 'N/A')[:50]}")
                print(f"    예산: {doc.metadata.get('price', 'N/A')}")
