# ==========================================================================================================
# Budget Filter Module
# ==========================================================================================================
# 용도: 사업 금액 조건으로 문서 필터링
# 지원: 억, 천만, 백만 단위 파싱
# 연산자: 이상, 이하, 초과
# 예시: "1억 이하", "5천만원 이상"
# ==========================================================================================================

import re


def parse_budget_from_query(query):
    """
    질의에서 금액 및 연산자 추출
    
    Args:
        query (str): 사용자 질의
    
    Returns:
        tuple: (금액(int), 연산자(str))
               예: (100000000, '이하')
    """
    q = query.replace(',', '').replace(' ', '')  # 쉼표, 공백 제거
    amount = 0
    
    # ========================================
    # [1] 금액 파싱
    # ========================================
    # 억
    if m := re.search(r'(\d+)억', q):
        amount += int(m.group(1)) * 100_000_000
    
    # 천만 (억 다음에 올 수 있음: "1억5천")
    if m := re.search(r'(?:억)?(\d+)천', q):
        amount += int(m.group(1)) * 10_000_000
    
    # 백만
    if m := re.search(r'(\d+)백만', q):
        amount += int(m.group(1)) * 1_000_000
    
    # ========================================
    # [2] 연산자 파싱
    # ========================================
    operator = '이하'  # 기본값
    if any(x in q for x in ['이상', '초과', '넘는']):
        operator = '이상'
    
    return amount, operator


def filter_by_budget(docs, query):
    """
    금액 조건으로 문서 필터링
    
    Args:
        docs (List[Document]): 문서 리스트
        query (str): 사용자 질의
    
    Returns:
        List[Document]: 필터링된 문서 리스트
    """
    amount, operator = parse_budget_from_query(query)
    
    # 금액 조건 없으면 원본 반환
    if amount == 0:
        return docs
    
    filtered = []
    for doc in docs:
        budget_num = doc.metadata.get('사업금액_num', 0)
        
        # [중요] 1원 이하(수의계약)는 금액 필터에서 제외
        # (특수 케이스이므로 별도 필터로 처리)
        if budget_num <= 1:
            continue
        
        # 조건 체크
        if operator == '이하' and budget_num <= amount:
            filtered.append(doc)
        elif operator == '이상' and budget_num >= amount:
            filtered.append(doc)
    
    if filtered:
        print(f"   [Budget Filter] '{amount:,}원 {operator}' → {len(filtered)}개")
    
    return filtered if filtered else docs


if __name__ == "__main__":
    # 테스트 코드
    from data.loader import load_documents
    
    docs = load_documents()
    
    # 테스트 쿼리
    test_queries = [
        "1억 이하",
        "5천만원 이상",
        "2억5천 이하",
        "1억5천만원"
    ]
    
    for q in test_queries:
        print(f"\n쿼리: {q}")
        amount, op = parse_budget_from_query(q)
        print(f"파싱: {amount:,}원 {op}")
        
        filtered = filter_by_budget(docs, q)
        print(f"결과: {len(filtered)}개")
