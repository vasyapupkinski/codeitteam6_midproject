# ==========================================================================================================
# Vector Store Loader Module
# ==========================================================================================================
# 용도: 기존 Qdrant DB 로드 (서비스 시작 시)
# 입력: 없음 (DB 파일 경로만)
# 출력: QdrantVectorStore (검색 가능 상태)
# ==========================================================================================================

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from config.settings import QDRANT_PATH, COLLECTION_NAME


def load_qdrant_db(qdrant_path=None, collection_name=None, embedding_function=None):
    """
    기존 Qdrant Vector DB 로드
    
    Args:
        qdrant_path (str): DB 경로 (기본값: settings.QDRANT_PATH)
        collection_name (str): 컬렉션 이름 (기본값: settings.COLLECTION_NAME)
        embedding_function: 임베딩 함수 (검색 시 필요)
    
    Returns:
        QdrantVectorStore: 로드된 벡터 스토어
    """
    if qdrant_path is None:
        qdrant_path = QDRANT_PATH
    if collection_name is None:
        collection_name = COLLECTION_NAME
    
    print(f"\nQdrant DB 로드 중...")
    print(f"  경로: {qdrant_path}")
    print(f"  컬렉션: {collection_name}")
    
    # ========================================
    # [1] 클라이언트 초기화
    # ========================================
    client = QdrantClient(path=qdrant_path)
    
    # ========================================
    # [2] 컬렉션 존재 확인
    # ========================================
    if not client.collection_exists(collection_name):
        raise FileNotFoundError(
            f"컬렉션 '{collection_name}'을 찾을 수 없습니다. "
            f"먼저 vectorstore/builder.py를 실행하여 DB를 생성하세요."
        )
    
    # ========================================
    # [3] Vector Store 초기화
    # ========================================
    # embedding_function 없이도 로드는 가능 (검색 시 필요)
    if embedding_function is None:
        print("  경고: embedding_function이 제공되지 않았습니다. 검색 기능 사용 불가.")
    
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_function,
    )
    
    # ========================================
    # [4] 컬렉션 정보 확인
    # ========================================
    collection_info = client.get_collection(collection_name)
    
    # qdrant-client 버전 호환성
    try:
        # 새 버전 (1.16+)
        vectors_count = collection_info.vectors_count
    except AttributeError:
        # 이전 버전 또는 다른 형식
        try:
            vectors_count = collection_info.points_count
        except:
            vectors_count = "Unknown"
    
    print(f"✓ DB 로드 완료!")
    print(f"  벡터 개수: {vectors_count}")
    
    return qdrant_store


if __name__ == "__main__":
    # 테스트 코드
    from embedding.bge_embedder import BGEEmbedder
    
    print("임베딩 모델 로드 중...")
    embedder = BGEEmbedder()
    
    print("\nVector DB 로드 중...")
    vector_store = load_qdrant_db(embedding_function=embedder.get_embedding_function())
    
    print("\n테스트 검색...")
    results = vector_store.similarity_search("학사정보시스템 구축", k=5)
    print(f"\n검색 결과: {len(results)}개")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.metadata.get('title', 'N/A')[:50]}")
