# ==========================================================================================================
# Reranker Module
# ==========================================================================================================
# 용도: CrossEncoder를 사용한 재순위화 (Reranking)
# 모델: BAAI/bge-reranker-v2-m3
# 최적화: 상위 N개만 리랭킹 (TOP_K_RERANK)
# ==========================================================================================================

import torch
from sentence_transformers import CrossEncoder
from config.settings import RERANKER_MODEL, RERANKER_DEVICE, TOP_K_RERANK


class Reranker:
    """
    CrossEncoder 기반 재순위화 클래스
    """
    
    def __init__(self, model_name=None, device=None):
        """
        Args:
            model_name (str): HuggingFace 모델명 (기본값: settings.RERANKER_MODEL)
            device (str): 'cuda' 또는 'cpu' (기본값: settings.RERANKER_DEVICE)
        """
        if model_name is None:
            model_name = RERANKER_MODEL
        if device is None:
            device = RERANKER_DEVICE
        
        print(f"Reranker 모델 로드 중... (device: {device})")
        self.model = CrossEncoder(model_name, device=device)
        print(f"✓ Reranker 로드 완료: {model_name}")
    
    def rerank(self, query, docs, top_k=None):
        """
        문서 재순위화
        
        Args:
            query (str): 검색 쿼리
            docs (List[Document]): 문서 리스트
            top_k (int): 리랭킹할 최대 문서 수 (기본값: settings.TOP_K_RERANK)
        
        Returns:
            List[tuple]: [(Document, score), ...] 점수 높은 순으로 정렬
        """
        if top_k is None:
            top_k = TOP_K_RERANK
        
        # ========================================
        # [최적화] 상위 N개만 리랭킹
        # ========================================
        docs_to_rerank = docs[:top_k] if len(docs) > top_k else docs
        
        # ========================================
        # [1] (쿼리, 문서) 쌍 생성
        # ========================================
        pairs = []
        for doc in docs_to_rerank:
            # 제목 + 본문 결합
            doc_text = f"제목: {doc.metadata.get('title', '')}\n내용: {doc.page_content}"
            pairs.append([query, doc_text])
        
        # ========================================
        # [2] CrossEncoder 점수 계산
        # ========================================
        scores = self.model.predict(pairs)
        
        # ========================================
        # [3] (문서, 점수) 튜플로 결합 및 정렬
        # ========================================
        scored_docs = list(zip(docs_to_rerank, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)  # 점수 높은 순
        
        return scored_docs


if __name__ == "__main__":
    # 테스트 코드
    from data.loader import load_documents
    from embedding.bge_embedder import BGEEmbedder
    from vectorstore.loader import load_qdrant_db
    from retrieval.dense_retriever import create_dense_retriever
    
    # 데이터 & 검색
    embedder = BGEEmbedder()
    vector_store = load_qdrant_db(embedding_function=embedder.get_embedding_function())
    retriever = create_dense_retriever(vector_store, k=10)
    
    query = "학사정보시스템"
    docs = retriever.invoke(query)
    print(f"\n검색 결과: {len(docs)}개")
    
    # Reranking
    reranker = Reranker()
    scored_docs = reranker.rerank(query, docs, top_k=5)
    
    print(f"\nReranking 후 (Top 5):")
    for i, (doc, score) in enumerate(scored_docs):
        print(f"{i+1}. [{score:.4f}] {doc.metadata.get('title', 'N/A')[:50]}")
