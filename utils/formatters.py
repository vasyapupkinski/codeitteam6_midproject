# ==========================================================================================================
# Document Formatter Module
# ==========================================================================================================
# 용도: 검색 결과를 LLM이 이해하기 쉬운 형식으로 변환
# 최적화: 각 문서당 400자로 제한 (settings.CHUNK_PREVIEW_LENGTH)
# ==========================================================================================================

from config.settings import CHUNK_PREVIEW_LENGTH


def format_docs_for_llm(docs):
    """
    검색 결과를 LLM 프롬프트용 텍스트로 변환
    
    Args:
        docs (List[tuple] or List[Document]): 검색 결과
            - List[tuple]: [(Document, score), ...] (Reranking 후)
            - List[Document]: [Document, ...] (일반 검색)
    
    Returns:
        str: 포맷팅된 컨텍스트 문자열
    """
    # ========================================
    # [1] 헤더 (문서 개수 명시)
    # ========================================
    context_string = f"총 {len(docs)}개 문서가 검색되었습니다.\n\n"
    
    for i, item in enumerate(docs):
        # ========================================
        # [2] Document 객체 추출
        # ========================================
        if isinstance(item, tuple):
            doc = item[0]  # (Document, score) 튜플에서 Document 추출
        else:
            doc = item  # 이미 Document
        
        # ========================================
        # [3] 메타데이터 추출
        # ========================================
        title = doc.metadata.get('title', '제목 미상')
        agency = doc.metadata.get('agency', '미상')
        folder_name = doc.metadata.get('source', '미상')
        chunk_id = doc.metadata.get('chunk_id', '')
        date = doc.metadata.get('pub_date', '미상')
        end_date = doc.metadata.get('end_date', '미상')
        
        # [핵심] Step 4에서 처리된 값 그대로 사용
        page_no = doc.metadata.get('page_no', '미상')
        price_str = doc.metadata.get('price', '미정')
        
        # ========================================
        # [4] 본문 추출 (동적 길이 조정)
        # ========================================
        chunk_text = doc.metadata.get('chunk_text', doc.page_content)
        
        # 기본값
        preview_len = CHUNK_PREVIEW_LENGTH
        
        # 동적 길이 로직 (Rank & Score 기반)
        if isinstance(item, tuple):
            score = item[1]
            rank = i + 1
            
            if rank <= 3:      # 상위 1~3위는 매우 길게
                preview_len = 1500
            elif score > 0.8:  # 유사도 매우 높음
                preview_len = 1000
            elif score > 0.5:  # 유사도 보통
                preview_len = 600
            else:              # 유사도 낮음
                preview_len = 400
        elif i < 3:            # 점수 없으면 순위만 고려
            preview_len = 1500
            
        chunk_preview = chunk_text[:preview_len]
        
        # ========================================
        # [5] 구조화된 텍스트 생성
        # ========================================
        context_string += f"""
[문서 {i+1}]
- 사업명: {title}
- 발주기관: {agency}
- 청크 ID: {chunk_id}
- ★공고번호: {page_no}
- 공고폴더: {folder_name}
- ★예산(사업금액): {price_str}
- ★입찰 마감일: {end_date}
- ★공개/공고 시기: {date}
- 본문 발췌: 
{chunk_preview}  (내용 생략됨...)
------------------------------------------
"""
    
    return context_string


if __name__ == "__main__":
    # 테스트 코드
    from data.loader import load_documents
    
    docs = load_documents()
    sample_docs = docs[:3]
    
    formatted = format_docs_for_llm(sample_docs)
    print(formatted)
    print(f"\n총 길이: {len(formatted)}자")
