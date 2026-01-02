# ==========================================================================================================
# Vector Store Builder Module
# ==========================================================================================================
# 용도: Qdrant Vector DB 생성 (GCP에서 1회 실행)
# 입력: List[Document] (data.loader에서)
# 출력: Qdrant DB (저장됨)
# ==========================================================================================================

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

from config.settings import (
    QDRANT_PATH,
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE
)


def build_qdrant_db(documents, embedding_function, qdrant_path=None, collection_name=None):
    """
    Qdrant Vector DB 생성
    
    Args:
        documents (List[Document]): LangChain Document 객체 리스트
        embedding_function: 임베딩 함수 (BGEEmbedder.get_embedding_function())
        qdrant_path (str): DB 저장 경로 (기본값: settings.QDRANT_PATH)
        collection_name (str): 컬렉션 이름 (기본값: settings.COLLECTION_NAME)
    
    Returns:
        QdrantVectorStore: 생성된 벡터 스토어
    """
    if qdrant_path is None:
        qdrant_path = QDRANT_PATH
    if collection_name is None:
        collection_name = COLLECTION_NAME
    
    print(f"\n{'='*60}")
    print(f" Qdrant Vector DB 생성 시작")
    print(f"{'='*60}")
    print(f"  경로: {qdrant_path}")
    print(f"  컬렉션: {collection_name}")
    print(f"  문서 개수: {len(documents)}")
    print(f"  차원: {EMBEDDING_DIMENSION}")
    
    # ========================================
    # [1] Qdrant 클라이언트 생성
    # ========================================
    client = QdrantClient(path=qdrant_path)
    
    # ========================================
    # [2] 기존 컬렉션 삭제 (재생성)
    # ========================================
    if client.collection_exists(collection_name):
        print(f"\n  기존 컬렉션 '{collection_name}' 삭제 중...")
        client.delete_collection(collection_name)
        print(f"  ✓ 삭제 완료")
    
    # ========================================
    # [3] 새 컬렉션 생성
    # ========================================
    print(f"\n  새 컬렉션 생성 중...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,  # 1024 (BGE-M3)
            distance=Distance.COSINE   # Cosine similarity
        ),
    )
    print(f"  ✓ 컬렉션 생성 완료")
    
    # ========================================
    # [4] Vector Store 초기화
    # ========================================
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_function,
    )
    
    # ========================================
    # [5] 문서 배치 삽입
    # ========================================
    print(f"\n  문서 임베딩 및 삽입 중 (배치 크기: {EMBEDDING_BATCH_SIZE})...")
    
    batch_size = EMBEDDING_BATCH_SIZE
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(documents), batch_size), total=total_batches, desc="  진행"):
        batch = documents[i : i + batch_size]
        qdrant_store.add_documents(batch)
    
    print(f"\n✓ Vector DB 생성 완료!")
    print(f"  총 {len(documents)}개 문서 삽입됨")
    print(f"{'='*60}\n")
    
    return qdrant_store


if __name__ == "__main__":
    # 테스트 코드
    from data.loader import load_documents
    from embedding.bge_embedder import BGEEmbedder
    
    print("데이터 로드 중...")
    documents = load_documents()
    
    print("임베딩 모델 로드 중...")
    embedder = BGEEmbedder()
    
    print("Vector DB 생성 중...")
    vector_store = build_qdrant_db(
        documents=documents,
        embedding_function=embedder.get_embedding_function()
    )
    
    print("테스트 검색...")
    results = vector_store.similarity_search("학사정보시스템", k=3)
    print(f"\n검색 결과: {len(results)}개")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.metadata.get('title', 'N/A')}")
