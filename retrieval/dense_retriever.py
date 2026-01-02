# ==========================================================================================================
# Dense Retriever Module (Qdrant)
# ==========================================================================================================
# 용도: Qdrant Vector DB를 사용한 Dense Retrieval
# 특징: 의미론적 유사도 기반 검색
# ==========================================================================================================

from config.settings import TOP_K_RETRIEVAL


def create_dense_retriever(vector_store, k=None):
    """
    Qdrant Dense Retriever 생성
    
    Args:
        vector_store (QdrantVectorStore): Qdrant 벡터 스토어
        k (int): 검색할 문서 개수 (기본값: settings.TOP_K_RETRIEVAL)
    
    Returns:
        Retriever: LangChain Retriever 객체
    """
    if k is None:
        k = TOP_K_RETRIEVAL
    
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    print(f"✓ Dense Retriever 생성 완료 (k={k})")
    return retriever


if __name__ == "__main__":
    # 테스트 코드
    from embedding.bge_embedder import BGEEmbedder
    from vectorstore.loader import load_qdrant_db
    
    embedder = BGEEmbedder()
    vector_store = load_qdrant_db(embedding_function=embedder.get_embedding_function())
    
    retriever = create_dense_retriever(vector_store)
    
    results = retriever.invoke("학사정보시스템")
    print(f"\n검색 결과: {len(results)}개")
    for i, doc in enumerate(results[:5]):
        print(f"{i+1}. {doc.metadata.get('title', 'N/A')[:50]}")
