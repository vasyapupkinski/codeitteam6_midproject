# ==========================================================================================================
# Hybrid Retriever Module (Ensemble)
# ==========================================================================================================
# 용도: Dense + Sparse Retrieval 결합
# 특징: 의미론적 검색 + 키워드 검색의 장점 결합
# 가중치: Dense 0.7 + Sparse 0.3
# ==========================================================================================================

from langchain_classic.retrievers import EnsembleRetriever
from config.settings import WEIGHT_DENSE, WEIGHT_SPARSE


def create_hybrid_retriever(dense_retriever, sparse_retriever):
    """
    Hybrid Retriever 생성 (Dense + Sparse)
    
    Args:
        dense_retriever: Qdrant Dense Retriever
        sparse_retriever: BM25 Sparse Retriever
    
    Returns:
        EnsembleRetriever: 하이브리드 Retriever
    """
    ensemble_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[WEIGHT_SPARSE, WEIGHT_DENSE]  # [0.3, 0.7]
    )
    
    print(f"✓ Hybrid Retriever 생성 완료 (Dense: {WEIGHT_DENSE}, Sparse: {WEIGHT_SPARSE})")
    return ensemble_retriever


if __name__ == "__main__":
    # 테스트 코드
    from data.loader import load_documents
    from embedding.bge_embedder import BGEEmbedder
    from vectorstore.loader import load_qdrant_db
    from retrieval.dense_retriever import create_dense_retriever
    from retrieval.sparse_retriever import create_sparse_retriever
    
    # 데이터 & 임베딩
    documents = load_documents()
    embedder = BGEEmbedder()
    vector_store = load_qdrant_db(embedding_function=embedder.get_embedding_function())
    
    # Dense & Sparse
    dense = create_dense_retriever(vector_store)
    sparse = create_sparse_retriever(documents)
    
    # Hybrid
    hybrid = create_hybrid_retriever(dense, sparse)
    
    # 검색 테스트
    results = hybrid.invoke("학사정보시스템")
    print(f"\n검색 결과: {len(results)}개")
    for i, doc in enumerate(results[:5]):
        print(f"{i+1}. {doc.metadata.get('title', 'N/A')[:50]}")
