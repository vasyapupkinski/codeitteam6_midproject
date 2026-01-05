# codeitteam6_midproject 6팀
- 이승완(팀장)
- 김모건
- 윤재형
- 김모건

# 입찰 공고 분석 RAG 시스템

공공기관 입찰 공고 문서를 분석하여 입찰 전략을 제공하는 AI 컨설턴트 시스템

## 프로젝트 개요

나라장터 입찰 공고 문서(제안요청서, 입찰공고서, 과업지시서)를 수집하여, 기업의 입찰 준비를 돕는 RAG(Retrieval-Augmented Generation) 기반 질의응답 시스템을 구축했습니다.

**목표**: 수백 페이지 분량의 복잡한 입찰 문서에서 핵심 정보를 신속하게 추출하고, 전문 컨설턴트 수준의 전략적 분석을 제공

---
<img width="653" height="1256" alt="image" src="https://github.com/user-attachments/assets/f8ae4412-88a3-4330-b55f-afea90417973" />


## 개발 과정 (7단계)

### 1단계: CSV 데이터 정제 및 EDA (5-7일)

**목표**: 깨끗하고 신뢰할 수 있는 메타데이터 확보

- **문제점**: 나라장터 제안요청서만으로는 정보 부족 (예산, 마감일 등 누락)
- **해결**:
  - 각 사업별로 **입찰공고서 + 과업지시서** 추가 수집
  - 공고번호, 사업명, 발주기관, 예산, 마감일 등 필수 메타데이터 수작업 검증
  - 수의계약(예산 0원/1원) 케이스 별도 처리
- **결과**: 신뢰도 높은 CSV 메타데이터 완성

### 2단계: 문서 파싱 (Upstage Document Parse API)

**목표**: PDF를 마크다운 형식으로 변환하여 구조화된 텍스트 추출

- **문제점**: Upstage 파서가 헤더(제목) 자동 인식 실패
- **해결**:
  - 문서 내 **헤더 패턴 분석** (정규식으로 "# 1. 사업개요", "## 가. 목적" 등 매칭)
  - 헤더를 수동으로 탐지하여 **계층 구조가 있는 마크다운** 생성
- **결과**: 문서 구조를 보존한 마크다운 텍스트

### 3단계: 파싱 결과 EDA 및 청킹 전략 수립

**목표**: 검색 효율을 높이기 위한 최적 청킹 방법 결정

- **분석 내용**:
  - 문서 길이 분포, 섹션별 문자 수, 헤더 깊이 등 통계 철저히 EDA 분석
  - 청킹 크기별 검색 성능 테스트
- **EDA 기반 결정**:
  - **MarkdownHeaderTextSplitter** (헤더 기반 분할)
    +
  - **RecursiveCharacterTextSplitter** (chunk_size=1500, overlap=500)
- **이유**: 헤더를 보존하면 문맥 유지가 가능하고, 적절한 오버랩으로 정보 단절 방지

### 4단계: 텍스트 이원화 (검색용 vs LLM용)

**목표**: 검색과 생성 각각에 최적화된 텍스트 제공

- **검색용 텍스트 (`search_text`)**:
  - 특수문자 제거, 평문화 → 임베딩 모델에 노이즈 최소화
  - 예: `| 항목 | 값 |` → `항목 값`
- **LLM용 텍스트 (`chunk_text`)**:
  - 원본 마크다운 구조 유지 (표, 목록 등)
  - LLM이 구조를 이해하고 정확한 답변 생성 가능
- **결과**: 검색 정확도와 답변 품질 동시 향상

### 5단계: Hybrid Search + 필터링 + Reranking

**검색 파이프라인 구성**:

```
1. Hybrid Retrieval (Dense + Sparse)
   - Dense: Qdrant (OpenAI text-embedding-3-small)
   - Sparse: BM25
   - Weight: 0.7 (Dense) + 0.3 (Sparse)
   
2. 고급 필터링 (Sequential)
   ├─ 날짜 필터: 연도/월/상반기/하반기/분기 자동 탐지
   ├─ 금액 필터: "5천만원 이상", "1억 이하" 파싱
   └─ 수의계약 필터: 특수 케이스 별도 처리

3. Reranking
   - CrossEncoder (BAAI/bge-reranker-v2-m3)
   - 의미적 유사도 재평가

4. Diversity Filter
   - 중복 프로젝트 제거 (project_id 기준)
```

### 6단계: LLM 답변 생성 (메타데이터 주입)

**답변 생성 최적화**:

- **메타데이터 우선 신뢰**:
  - ★표시된 메타데이터(예산, 마감일, 공고번호)를 본문보다 우선
  - 할루시네이션 방지
- **페르소나 설정**:
  - "20년 차 입찰 전문 컨설턴트"
  - 전략적 분석(입찰 전략, 위험 요소, 실무 질의서) 포함
- **모델 선택**:
  - 초기 실험: Gemma 2 9B, Qwen 2.5 7B
  - **최종 선택: OpenAI GPT-5-mini** (한국어 품질 및 컨설팅 톤 우수)

### 7단계: LangGraph + Gradio UI

**고급 기능 구현**:

- **LangGraph Agent**:
  - 로컬 검색 1차 → 문서 평가 → 부족 시 질문 재작성 후 2차 검색
  - 웹 검색 (DuckDuckGo) 폴백
  - 날짜 계산 툴 (마감일까지 일수 등)
- **Gradio UI**:
  - 실시간 스트리밍 답변
  - 검색된 문서 출처 표시

---

## 프로젝트 구조

```
시나리오 A project py/
├── agent/                    # LangGraph Agent 구현
│   ├── graph.py             # Agent 그래프 정의
│   ├── nodes.py             # Agent 노드 (검색, 평가, 재작성, 생성)
│   ├── state.py             # Agent 상태 관리
│   └── tools.py             # Agent 도구 (날짜 계산 등)
│
├── config/                   # 설정 관리
│   ├── __init__.py
│   └── settings.py          # 하이퍼파라미터, 모델 경로
│
├── data/                     # 데이터 로더
│
├── embedding/                # 임베딩 모듈 (OpenAI Embeddings)
│
├── filtering/                # 필터링 모듈
│                            # - 날짜 필터 (연도/월/분기/반기)
│                            # - 금액 필터 (억/천만 파싱)
│                            # - 수의계약 필터
│
├── llm/                      # LLM 래퍼 (OpenAI GPT-5-mini)
│
├── reranking/                # Reranking 모듈 (CrossEncoder)
│
├── retrieval/                # 검색 모듈
│                            # - Dense (Qdrant)
│                            # - Sparse (BM25)
│                            # - Ensemble (Hybrid)
│
├── scripts/                  # 유틸리티 스크립트
│   └── build_db.py          # Vector DB 생성
│
├── utils/                    # 공통 유틸리티
│
├── vectorstore/              # Qdrant 벡터스토어 관리
│
├── web_search/               # 웹 검색 (DuckDuckGo)
│
├── 시나리오 B 완성된 쥬피터 파일들/  # 노트북 원본
│
├── .gitignore                # Git 제외 파일 목록
├── app_simple.py             # 단순 Gradio UI (Agent 미사용)
├── dependencies.py           # 공통 의존성 임포트
├── main.py                   # Gradio UI + LangGraph Agent
└── requirements.txt          # Python 패키지 목록
```

---

## 빠른 시작

### 1. 환경 설정

```bash
# Python 3.10+ 권장
pip install -r requirements.txt

# .env 파일에 API 키 설정
OPENAI_API_KEY=your_key_here
```

### 2. Vector DB 생성 (1회만)

```bash
python scripts/build_db.py
```

### 3. Gradio 앱 실행

```bash
python main.py
```

브라우저에서 `http://localhost:7860` 접속

---

## 핵심 설계 철학

| 단계 | 철학 | 이유 |
|------|------|------|
| CSV 정제 | Garbage In, Garbage Out | 메타데이터 품질이 전체 시스템의 신뢰도를 결정 |
| 텍스트 이원화 | 검색과 생성은 전혀 다른 요구사항 | 각 단계에 최적화된 텍스트로 성능 극대화 |
| 필터링 우선 | Reranking 전 노이즈 제거 | 불필요한 연산 & 중복 감소(제거) 및 정확도 향상 |
| 메타데이터 주입 | 구조화된 정보 우선 | LLM 환각 방지 및 사실 기반 답변 보장 |

---

## 예시 질의

```
 "2024년 상반기 학사정보시스템 구축 사업 중 예산 상위 3개 추천해줘"
 "수의계약 건들 중에서 1억원 이하 보안 시스템 사업"
 "재난안전 시스템 관련 사업의 핵심 요구사항 비교해줘"
 "고려대학교가 발주한 차세대 포털 시스템의 입찰 전략 제시해줘"
```

---

## 트러블슈팅

### OpenAI API 키 오류
```bash
# .env 파일 생성
echo "OPENAI_API_KEY=sk-..." > .env
```

### Qdrant DB 없음
```bash
python scripts/build_db.py  # DB 재생성
```
---

## 라이선스

MIT License

## 개발팀

코드잇 6팀

코드잇 AI 중급 프로젝트 6팀 협업일지
- 이승완(팀장) https://foremost-andesaurus-a63.notion.site/2c6bfba1db0b80a39e83e4d2bc2c1a17?source=copy_link
- 김모건 https://peach-antimatter-e36.notion.site/_-_-2c4068e43a16802d9ae1fb419238aaac?source=copy_link
- 윤재형 https://www.notion.so/251211-2c68892abe41814883d2eb097b135239?source=copy_link
- 김승우 https://www.notion.so/Daily-12-12-2de289f7446a811f9128c34daf5e3be3

최종 보고서(총 2개)
- [중급 프로젝트 보고서 1 다운로드](./report/codeitteam6_6팀_중급_프로젝트_보고서.pdf)
- [중급 프로젝트 보고서 2] https://drive.google.com/file/d/1gTR_FCoPaGEcgIRNLEGInhcqEyrqnW2Y/view?usp=drive_link
