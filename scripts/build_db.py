# ==========================================================================================================
# Build Vector DB Script
# ==========================================================================================================
# 용도: Qdrant Vector DB 생성 (GCP에서 1회 실행)
# 흐름: CSV 로드 → BGE-M3 임베딩 → Qdrant DB 저장
# 사용법: python scripts/build_db.py
# ==========================================================================================================

import sys
import os

# 프로젝트 루트를 PATH에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_documents
from embedding.bge_embedder import BGEEmbedder
from embedding.openai_embedder import OpenAIEmbedder
from vectorstore.builder import build_qdrant_db
from config.settings import EMBEDDING_MODEL


def main():
    """
    Vector DB 생성 메인 함수
    """
    print("\n" + "="*80)
    print(" Qdrant Vector DB 생성 스크립트")
    print("="*80)
    
    # ========================================
    # [1] CSV 데이터 로드
    # ========================================
    print("\n[Step 1/3] CSV 데이터 로드 중...")
    documents = load_documents()
    
    # ========================================
    # [2] 임베딩 모델 로드
    # ========================================
    print(f"\n[Step 2/3] 임베딩 모델 로드 중... ({EMBEDDING_MODEL})")
    
    if "text-embedding" in EMBEDDING_MODEL:
        embedder = OpenAIEmbedder()
    else:
        embedder = BGEEmbedder()
    
    # ========================================
    # [3] Vector DB 생성
    # ========================================
    print("\n[Step 3/3] Vector DB 생성 중...")
    vector_store = build_qdrant_db(
        documents=documents,
        embedding_function=embedder.get_embedding_function()
    )
    
    # ========================================
    # [완료]
    # ========================================
    print("="*80)
    print(" ✓ 모든 작업 완료!")
    print("="*80)
    print("\n다음 단계:")
    print("  1. Gradio 앱 실행: python main.py")
    print("  2. 또는 GCP에 배포하여 서비스 시작")
    print()


if __name__ == "__main__":
    main()
