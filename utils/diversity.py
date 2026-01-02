# ==========================================================================================================
# Diversity Module
# ==========================================================================================================
# 용도: 중복 프로젝트 제거 (다양성 확보)
# 방법: project_id 기준으로 중복 제거, 점수 높은 것만 유지
# ==========================================================================================================

from config.settings import TOP_K_FINAL


def get_diverse_top_k(scored_docs, k=None):
    """
    중복 프로젝트 제거하면서 상위 K개 선택
    
    Args:
        scored_docs (List[tuple]): [(Document, score), ...] 점수로 정렬된 리스트
        k (int): 최종 선택할 문서 개수 (기본값: settings.TOP_K_FINAL)
    
    Returns:
        List[tuple]: 중복 제거된 상위 K개 [(Document, score), ...]
    """
    if k is None:
        k = TOP_K_FINAL
    
    final_results = []
    seen_projects = set()  # 이미 선택된 project_id 추적
    
    for doc, score in scored_docs:
        project_id = doc.metadata.get('project_id')
        
        # 이미 선택된 프로젝트면 스킵
        if project_id in seen_projects:
            continue
        
        # 선택
        final_results.append((doc, score))
        seen_projects.add(project_id)
        
        # K개 채우면 종료
        if len(final_results) >= k:
            break
    
    return final_results


if __name__ == "__main__":
    # 테스트 코드
    from data.loader import load_documents
    
    docs = load_documents()
    
    # 동일 프로젝트 문서 여러 개 생성 (테스트용)
    test_scored_docs = [
        (docs[0], 0.95),  # project_1
        (docs[1], 0.90),  # project_2
        (docs[0], 0.85),  # project_1 (중복)
        (docs[2], 0.80),  # project_3
        (docs[1], 0.75),  # project_2 (중복)
    ]
    
    print("원본 (5개):")
    for i, (doc, score) in enumerate(test_scored_docs):
        print(f"{i+1}. [{score:.2f}] {doc.metadata.get('project_id')} - {doc.metadata.get('title', 'N/A')[:30]}")
    
    diverse = get_diverse_top_k(test_scored_docs, k=3)
    
    print("\n다양성 필터 후 (Top 3):")
    for i, (doc, score) in enumerate(diverse):
        print(f"{i+1}. [{score:.2f}] {doc.metadata.get('project_id')} - {doc.metadata.get('title', 'N/A')[:30]}")
