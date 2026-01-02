# ==========================================================================================================
# RAG Components Initialization (Singleton)
# ==========================================================================================================
import sys
import os
from data.loader import load_documents
from embedding.bge_embedder import BGEEmbedder
from embedding.openai_embedder import OpenAIEmbedder
from vectorstore.loader import load_qdrant_db
from retrieval.dense_retriever import create_dense_retriever
from retrieval.sparse_retriever import create_sparse_retriever
from retrieval.hybrid_retriever import create_hybrid_retriever
from reranking.reranker import Reranker
from llm.gemma import GemmaLLM
from llm.qwen import QwenLLM
from llm.openai import OpenAILLM
from config.settings import EMBEDDING_MODEL

print(">>> [초기화] RAG 컴포넌트 로드 중...")

# 1. 문서 및 임베딩
documents = load_documents()

# 임베딩 모델 선택
if "text-embedding" in EMBEDDING_MODEL:
    embedder = OpenAIEmbedder()
else:
    embedder = BGEEmbedder()

vector_store = load_qdrant_db(embedding_function=embedder.get_embedding_function())

# 2. 검색기
dense = create_dense_retriever(vector_store)
sparse = create_sparse_retriever(documents)
hybrid_retriever = create_hybrid_retriever(dense, sparse)

# 3. Reranker
reranker = Reranker()

# 4. LLM 모델 (Lazy Loading - 사용 시점에 로드)
_llm_cache = {}

def get_llm(model_name):
    """
    LLM을 필요 시점에 로드 (메모리 절약)
    """
    if model_name not in _llm_cache:
        print(f"\n>>> [LLM 로드] {model_name} 초기화 중...")
        if model_name == "Gemma 2 9B":
            _llm_cache[model_name] = GemmaLLM()
        elif model_name == "Qwen 2.5 7B":
            _llm_cache[model_name] = QwenLLM()
        elif model_name == "GPT-5-mini":
            # 사용자 요청에 따라 'gpt-5-mini' 모델 ID 직접 사용
            _llm_cache[model_name] = OpenAILLM(model_name="gpt-5-mini")
        else:
            raise ValueError(f"Unknown model: {model_name}")
    return _llm_cache[model_name]

# 하위 호환성: llms dict 유지 (하지만 실제로는 lazy)
class LazyLLMDict:
    def __getitem__(self, key):
        return get_llm(key)
    
    def keys(self):
        return ["Gemma 2 9B", "Qwen 2.5 7B", "GPT-5-mini"]

llms = LazyLLMDict()

print(">>> [완료] 모든 컴포넌트 로드 완료 (LLM은 사용 시 로드)")
