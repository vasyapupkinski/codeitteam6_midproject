# ==========================================================================================================
# OpenAI Embedding Module
# ==========================================================================================================
# 모델: text-embedding-3-small
# 특징: 1536차원, OpenAI API 사용
# ==========================================================================================================

import os
from langchain_openai import OpenAIEmbeddings
from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSION

class OpenAIEmbedder:
    """
    OpenAI Embedding Class
    """
    
    def __init__(self, model_name=None, device=None):
        """
        Args:
            model_name (str): OpenAI 모델명 (기본값: text-embedding-3-small)
            device (str): 사용되지 않음 (API 기반)
        """
        if model_name is None:
            model_name = EMBEDDING_MODEL
            
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        
        print(f"OpenAI 임베딩 모델 로드 중... ({model_name})")
        
        self.model = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key,
            dimensions=EMBEDDING_DIMENSION
        )
        
        print(f"✓ 모델 로드 완료: {model_name}")
    
    def embed_documents(self, texts):
        return self.model.embed_documents(texts)
    
    def embed_query(self, text):
        return self.model.embed_query(text)
    
    def get_embedding_function(self):
        return self.model
