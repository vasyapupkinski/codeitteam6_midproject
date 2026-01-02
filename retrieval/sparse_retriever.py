# ==========================================================================================================
# Sparse Retriever Module (BM25)
# ==========================================================================================================
# 용도: 키워드 매칭 기반 검색 (BM25)
# 특징: Kiwi 형태소 분석기를 사용하여 한국어 토큰화
# ==========================================================================================================

from langchain_community.retrievers import BM25Retriever
from kiwipiepy import Kiwi
from config.settings import TOP_K_RETRIEVAL


def create_sparse_retriever(documents, k=None):
    """
    BM25 Sparse Retriever 생성
    
    Args:
        documents (List[Document]): 검색 대상 문서 리스트
        k (int): 검색할 문서 개수 (기본값: settings.TOP_K_RETRIEVAL)
    
    Returns:
        BM25Retriever: LangChain Retriever 객체
    """
    if k is None:
        k = TOP_K_RETRIEVAL
        
    print("Sparse Retriever (BM25) 생성 중... (형태소 분석: Kiwi)")
    
    # Kiwi 형태소 분석기 초기화
    kiwi = Kiwi()
    
    # 토크나이저 함수 정의 (인코딩 오류 방지)
    def kiwi_tokenize(text):
        try:
            # Windows 인코딩 문제 방지
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            elif not isinstance(text, str):
                text = str(text)
            
            # 유니코드 정규화
            import unicodedata
            text = unicodedata.normalize('NFC', text)
            
            return [token.form for token in kiwi.tokenize(text)]
        except Exception as e:
            # 오류 발생 시 공백 기준 분리
            print(f"Kiwi 토큰화 오류 (fallback 사용): {e}")
            return text.split()
    
    # BM25 Retriever 생성
    retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=kiwi_tokenize  # 한국어 형태소 분석 적용
    )
    retriever.k = k
    
    print(f"✓ Sparse Retriever 생성 완료 (k={k})")
    return retriever


if __name__ == "__main__":
    # 테스트 코드
    from data.loader import load_documents
    
    documents = load_documents()
    retriever = create_sparse_retriever(documents)
    
    results = retriever.invoke("학사정보시스템")
    print(f"\n검색 결과: {len(results)}개")
    for i, doc in enumerate(results[:5]):
        print(f"{i+1}. {doc.metadata.get('title', 'N/A')[:50]}")
