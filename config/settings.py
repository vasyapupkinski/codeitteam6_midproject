# ==========================================================================================================
# RAG Pipeline Configuration Settings
# ==========================================================================================================
# 용도: 모든 모듈에서 사용하는 설정값을 중앙에서 관리
# 변경 시: 이 파일만 수정하면 전체 시스템 설정이 변경됨
# ==========================================================================================================

import os
from dotenv import load_dotenv

# .env 파일 로드 (가장 먼저 실행)
load_dotenv()

# ==========================================
# [경로 설정] - 로컬/GCP 모두 호환
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CSV 데이터 (project py/data/ 폴더)
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "rag_chunks_recursive.csv")

# Vector DB (project py/local_qdrant_db/)
QDRANT_PATH = os.path.join(BASE_DIR, "local_qdrant_db")
COLLECTION_NAME = "rfp_recursive_search"

# ==========================================
# [모델 설정]
# ==========================================
# Embedding Model (HuggingFace or OpenAI)
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI 모델로 변경
EMBEDDING_DEVICE = "cpu"  # API는 device 불필요
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small dimension

# Reranker Model (HuggingFace)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE = "cuda"  # or "cpu"

# LLM Models (HuggingFace & OpenAI)
LLM_MODELS = {
    "Gemma 2 9B": "google/gemma-2-9b-it",
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct",
    "GPT-5-mini": "gpt-5-mini"
}
DEFAULT_LLM = "GPT-5-mini"
LLM_DEVICE = "cuda"  # or "cpu"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 8192

# ==========================================
# [검색 설정]
# ==========================================
# Retrieval
TOP_K_RETRIEVAL = 50  # Hybrid 검색 개수 (Dense + Sparse)
WEIGHT_DENSE = 0.7    # Dense 가중치
WEIGHT_SPARSE = 0.3   # Sparse 가중치

# Rerank 설정
TOP_K_RERANK = 30       # Rerank 후보 (Notebook Parity: Notebook은 검색된 50개 전체를 Rerank함)
TOP_K_FINAL = 7         # 최종 LLM 전달 (Notebook Parity: 5개)

# ==========================================
# [LLM 컨텍스트 설정]
# ==========================================
CHUNK_PREVIEW_LENGTH = 1500  # 각 문서당 LLM에 전달할 최대 글자 수

# ==========================================
# [배치 설정]
# ==========================================
EMBEDDING_BATCH_SIZE = 100  # 임베딩 배치 크기 (DB 생성 시)

# ==========================================
# [로깅 설정]
# ==========================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==========================================
# [Gradio 설정]
# ==========================================
GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7860