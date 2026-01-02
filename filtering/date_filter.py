# ==========================================================================================================
# Date Filter Module
# ==========================================================================================================
# 용도: 날짜 조건으로 문서 필터링
# 지원: 연도, 월, 상반기/하반기, 분기
# 예시: "2024년", "3월", "상반기", "2분기"
# ==========================================================================================================

import re


def filter_by_date(docs, query):
    """
    날짜 조건으로 문서 필터링
    
    Args:
        docs (List[Document]): 문서 리스트
        query (str): 사용자 질의
    
    Returns:
        List[Document]: 필터링된 문서 리스트
    """
    filtered = docs
    filter_applied = False
    
    # ========================================
    # [1] 연도 필터
    # ========================================
    years = re.findall(r'(20\d{2})년?', query)
    if years:
        target_year = years[0]
        temp_filtered = [
            doc for doc in filtered 
            if target_year in str(doc.metadata.get('end_date', '')) 
            or target_year in str(doc.metadata.get('pub_date', ''))
        ]
        if temp_filtered:
            filtered = temp_filtered
            filter_applied = True
            print(f"   [Date Filter] 연도 '{target_year}' 감지 → {len(filtered)}개 문서")
    
    # ========================================
    # [2] 월 필터
    # ========================================
    month_match = re.search(r'(\d{1,2})월', query)
    if month_match:
        target_month = month_match.group(1).zfill(2)  # 01, 02, ..., 12
        temp_filtered = [
            doc for doc in filtered 
            if f"-{target_month}" in str(doc.metadata.get('end_date', '')) 
            or f"{target_month}월" in str(doc.metadata.get('pub_date', ''))
        ]
        if temp_filtered:
            filtered = temp_filtered
            filter_applied = True
            print(f"   [Date Filter] 월 '{target_month}월' 감지 → {len(filtered)}개 문서")
    
    # ========================================
    # [3] 반기 필터
    # ========================================
    if '상반기' in query:
        temp_filtered = [
            doc for doc in filtered 
            if any(
                f"-{m:02d}" in str(doc.metadata.get('end_date', ''))
                or f"{m}월" in str(doc.metadata.get('pub_date', ''))
                for m in range(1, 7)  # 1~6월
            )
        ]
        if temp_filtered:
            filtered = temp_filtered
            filter_applied = True
            print(f"   [Date Filter] '상반기' 감지 → {len(filtered)}개 문서")
    
    elif '하반기' in query:
        temp_filtered = [
            doc for doc in filtered 
            if any(
                f"-{m:02d}" in str(doc.metadata.get('end_date', ''))
                or f"{m}월" in str(doc.metadata.get('pub_date', ''))
                for m in range(7, 13)  # 7~12월
            )
        ]
        if temp_filtered:
            filtered = temp_filtered
            filter_applied = True
            print(f"   [Date Filter] '하반기' 감지 → {len(filtered)}개 문서")
    
    # ========================================
    # [4] 분기 필터
    # ========================================
    quarter_match = re.search(r'([1-4])분기', query)
    if quarter_match:
        q = int(quarter_match.group(1))
        q_map = {
            1: range(1, 4),   # 1~3월
            2: range(4, 7),   # 4~6월
            3: range(7, 10),  # 7~9월
            4: range(10, 13)  # 10~12월
        }
        q_range = q_map[q]
        
        temp_filtered = [
            doc for doc in filtered 
            if any(
                f"-{m:02d}" in str(doc.metadata.get('end_date', ''))
                or f"{m}월" in str(doc.metadata.get('pub_date', ''))
                for m in q_range
            )
        ]
        if temp_filtered:
            filtered = temp_filtered
            filter_applied = True
            print(f"   [Date Filter] '{q}분기' 감지 → {len(filtered)}개 문서")
    
    # 필터 적용되지 않았으면 원본 반환
    return filtered if filter_applied else docs


if __name__ == "__main__":
    # 테스트 코드
    from data.loader import load_documents
    
    docs = load_documents()
    
    # 테스트 쿼리
    test_queries = [
        "2024년 사업",
        "3월 공고",
        "상반기 프로젝트",
        "2분기 입찰"
    ]
    
    for q in test_queries:
        print(f"\n쿼리: {q}")
        filtered = filter_by_date(docs, q)
        print(f"결과: {len(filtered)}개")
