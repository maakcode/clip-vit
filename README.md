# CLIP을 이용한 이미지 검색

Streamlit과 OpenAI의 CLIP 모델을 사용하여 텍스트로 유사 이미지를 검색하는 웹 애플리케이션입니다.

<img src="https://raw.githubusercontent.com/maakcode/clip-vit/refs/heads/main/docs/1.avif" width="600" />

## 주요 기능

- 여러 이미지 파일을 한 번에 업로드
- FAISS를 사용하여 이미지 검색 인덱스 구축
- 자연어(텍스트)를 이용한 이미지 검색
- 실시간 유사도 검색 결과 확인
- LLMTranslator를 활용한 다국어 텍스트 쿼리 지원 (ibm-granite/granite-4.0-h-micro 모델 사용)

## 동작 원리

이 애플리케이션은 CLIP(Contrastive Language-Image Pre-training) 모델을 사용하여 이미지와 텍스트 쿼리의 임베딩(embedding)을 생성합니다. 이 임베딩은 이미지와 텍스트의 의미적 내용을 고차원 벡터 공간에 표현한 것입니다.

생성된 이미지 임베딩은 효율적인 유사도 검색을 위해 FAISS(Facebook AI Similarity Search) 인덱스에 저장됩니다. 사용자가 텍스트 쿼리를 입력하면, LLMTranslator (ibm-granite/granite-4.0-h-micro 모델 기반)가 해당 쿼리를 영어로 번역한 후, 번역된 텍스트의 임베딩을 계산하고 FAISS 인덱스 내에서 가장 유사한 이미지 임베딩을 찾아 결과를 반환합니다.

## 개발 환경 설정 및 실행

이 프로젝트는 `uv`를 사용하여 가상 환경 및 패키지를 관리합니다.

### 1. 사전 준비

- Python 3.12 이상
- `uv` 설치

### 2. 애플리케이션 실행

`uv run`을 사용하면 가상 환경을 활성화하지 않고도 바로 실행할 수 있습니다.

```bash
uv run python -m streamlit run main.py
```

실행 후 터미널에 표시되는 로컬 URL(보통 `http://localhost:8501`)에 접속하여 애플리케이션을 사용할 수 있습니다.

---

**참고:**

- 앱을 처음 실행할 때 CLIP 모델을 다운로드하므로 시간이 다소 걸릴 수 있습니다.
- 현재 FAISS 인덱스는 메모리에서만 생성되므로 앱을 재시작하면 사라집니다.
- macOS에서 발생할 수 있는 `OMP: Error #15` 오류는 `main.py` 스크립트 내에서 관련 환경 변수를 설정하여 해결했습니다.

---

_이 프로젝트의 README 작성 및 코드 일부는 Google Gemini의 도움을 받아 작성되었습니다._
