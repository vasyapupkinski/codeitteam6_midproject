# ==========================================================================================================
# Data Loader Module
# ==========================================================================================================
# 용도: CSV 파일을 로드하고 LangChain Document 객체로 변환
# 입력: rag_chunks_recursive.csv
# 출력: List[Document] (메타데이터 포함)
# ==========================================================================================================

import os
import pandas as pd
from tqdm import tqdm
from langchain_core.documents import Document
from config.settings import CSV_PATH


def load_csv_data(csv_path=None):
    """
    CSV 파일 로드
    
    Args:
        csv_path (str): CSV 파일 경로 (기본값: settings.CSV_PATH)
    
    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    if csv_path is None:
        csv_path = CSV_PATH
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    
    df = pd.read_csv(csv_path).fillna("")
    print(f"✓ CSV 로드 완료: {len(df)}개 청크")
    
    return df


def convert_to_documents(df):
    """
    DataFrame을 LangChain Document 객체 리스트로 변환
    
    [중요] Step 4의 메타데이터 처리 로직을 그대로 적용:
    - 0원/1원 → "수의계약(예산 협상 대상)" + is_private_contract=True
    - 그 외 → CSV 값 신뢰 + is_private_contract=False
    
    Args:
        df (pd.DataFrame): CSV 데이터프레임
    
    Returns:
        List[Document]: Document 객체 리스트
    """
    def _classify_domain(text):
        """키워드 기반 도메인 분류"""
        text = text.lower()
        if any(w in text for w in ['대학', '학교', '학사', '교원', '학생', '교육', '강의', '이러닝', '캠퍼스']):
            return "교육"
        elif any(w in text for w in ['국방', '군', '부대', '방위', '사령부', '전력', '병무']):
            return "국방"
        elif any(w in text for w in ['은행', '금융', '보험', '증권', '카드', '금고']):
            return "금융"
        elif any(w in text for w in ['병원', '의료', '보건', '약무', '임상', '건강']):
            return "의료"
        elif any(w in text for w in ['시청', '도청', '공단', '재단', '공사', '정부', '부처', '진흥원', '연구원', '센터', '협회', '공항', '항공', '철도', '교통', '도시']):
            return "공공"
        else:
            return "기타"

    documents = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Document 변환"):
        # ========================================
        # [1] 공고번호
        # ========================================
        page_no = str(row.get('공고번호', '')).strip()
        if not page_no:
            page_no = str(row.get('page_no', ''))
        
        # ========================================
        # [2] 금액 및 수의계약 처리
        # ========================================
        try:
            budget_num = int(row.get('사업금액_num', 0))
        except:
            budget_num = 0
        
        # [규칙 적용]
        if budget_num <= 1:
            price_str = "수의계약(예산 협상 대상)"
            is_private = True
        else:
            # 원화 포맷팅 적용 (예: 1,000,000원)
            price_str = f"{budget_num:,}원"
            is_private = False
            
        # ========================================
        # [3] 도메인 분류
        # ========================================
        title = str(row.get('사업명', ''))
        agency = str(row.get('발주기관', ''))
        folder_name = str(row.get('folder_name', ''))
        # 제목 + 기관명 + 폴더명까지 모두 합쳐서 키워드 검사
        domain = _classify_domain(title + " " + agency + " " + folder_name)
        
        # ========================================
        # [4] 특이사항 텍스트
        # ========================================
        special_note = ""
        if is_private:
            special_note = "[특이사항: 본 공고는 수의시담 또는 직찰 건으로 예산이 수의계약(예산 협상 대상)입니다.]"
        
        # ========================================
        # [5] 본문 구성
        # ========================================
        # 타이틀와 도메인 정보를 본문 최상단에 추가
        header = f"[{domain} 분야] {title}"
        
        page_content = str(row.get('search_text', ''))
        page_content = f"{header}\n{page_content}"
        
        if special_note:
            page_content = f"{special_note}\n{page_content}"
        
        original_chunk = str(row.get('chunk_text', ''))
        if special_note:
            original_chunk = f"--- 공고 정보 ---\n분야: {domain}\n{special_note}\n----------------\n\n{original_chunk}"
        else:
            original_chunk = f"--- 공고 정보 ---\n분야: {domain}\n----------------\n\n{original_chunk}"
        
        # ========================================
        # [6] 메타데이터 저장
        # ========================================
        metadata = {
            "page_no": page_no,
            "price": price_str,
            "사업금액_num": budget_num,
            "domain": domain,  # 도메인 추가
            
            "chunk_id": row.get('chunk_id'),
            "project_id": str(row.get('project_id')),
            "source": row.get('folder_name'),
            "title": title,
            "agency": agency,
            "pub_date": str(row.get('공개_월', '')),
            "end_date": str(row.get('입찰마감일', '')),
            "h1": row.get('metadata_h1'),
            "chunk_text": original_chunk,
            "is_private_contract": is_private
        }
        
        documents.append(Document(page_content=page_content, metadata=metadata))
    
    print(f"✓ Document 변환 완료: {len(documents)}개")
    return documents


def load_documents(csv_path=None):
    """
    CSV 로드 + Document 변환 (통합 함수)
    
    Args:
        csv_path (str): CSV 파일 경로 (옵션)
    
    Returns:
        List[Document]: Document 객체 리스트
    """
    df = load_csv_data(csv_path)
    documents = convert_to_documents(df)
    return documents


if __name__ == "__main__":
    # 테스트 코드
    docs = load_documents()
    print(f"\n샘플 Document:")
    print(f"  page_content: {docs[0].page_content[:100]}...")
    print(f"  metadata: {docs[0].metadata}")
