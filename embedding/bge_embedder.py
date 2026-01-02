# ==========================================================================================================
# BGE-M3 Embedding Module
# ==========================================================================================================
# 용도: BAAI/bge-m3 모델을 사용한 문서 임베딩
# 특징: 
#   - 1024차원 (text-embedding-3-small의 1536차원 대비)
#   - Multilingual 특화 (한국어 성능 우수)
#   - HuggingFace 모델 (API 비용 없음)
# ==========================================================================================================

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL, EMBEDDING_DEVICE


class BGEEmbedder:
    """
    BGE-M3 임베딩 클래스
    
    OpenAI text-embedding-3-small을 대체하는 오픈소스 임베딩 모델
    """
    
    def __init__(self, model_name=None, device=None):
        """
        Args:
            model_name (str): HuggingFace 모델명 (기본값: BAAI/bge-m3)
            device (str): 'cuda' 또는 'cpu' (기본값: settings.EMBEDDING_DEVICE)
        """
        if model_name is None:
            model_name = EMBEDDING_MODEL
        if device is None:
            device = EMBEDDING_DEVICE
        
        print(f"BGE-M3 임베딩 모델 로드 중... (device: {device})")
        
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}  # 정규화 (cosine similarity 최적화)
        )
        
        print(f"✓ 모델 로드 완료: {model_name}")
    
    def embed_documents(self, texts):
        """
        문서 리스트 임베딩
        
        Args:
            texts (List[str]): 임베딩할 텍스트 리스트
        
        Returns:
            List[List[float]]: 임베딩 벡터 리스트 (각 벡터는 1024차원)
        """
        return self.model.embed_documents(texts)
    
    def embed_query(self, text):
        """
        단일 쿼리 임베딩
        
        Args:
            text (str): 임베딩할 쿼리 텍스트
        
        Returns:
            List[float]: 임베딩 벡터 (1024차원)
        """
        return self.model.embed_query(text)
    
    def get_embedding_function(self):
        """
        LangChain 호환 임베딩 함수 반환
        (QdrantVectorStore 등에서 사용)
        
        Returns:
            HuggingFaceEmbeddings: 임베딩 모델 객체
        """
        return self.model


if __name__ == "__main__":
    # 테스트 코드
    embedder = BGEEmbedder()
    
    # 단일 쿼리 테스트
    query = "한영대학교 학사정보시스템 구축"
    query_embedding = embedder.embed_query(query)
    print(f"\n쿼리 임베딩 차원: {len(query_embedding)}")
    print(f"샘플 값: {query_embedding[:5]}")
    
    # 문서 배치 테스트
    docs = ["문서 1", "문서 2", "문서 3"]
    doc_embeddings = embedder.embed_documents(docs)
    print(f"\n문서 임베딩 개수: {len(doc_embeddings)}")
    print(f"각 문서 차원: {len(doc_embeddings[0])}")
