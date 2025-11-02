# [1.0.0](https://github.com/maakcode/deep-image-search/releases/tag/1.0.0)

- 진행중인 개인 프로젝트에서 자연어로 이미지 내용을 묘사해서 이미지를 검색할 수 있는 기능의 기술 검증을 위해 프로젝트를 시작함.
- UI는 최대한 빠르게 구현 가능하면서도 유려해서 [Streamlit](https://streamlit.io) 선택함.
- 텍스트와 이미지 임베딩 모델은 원래는 [EVA-CLIP](https://github.com/baaivision/EVA) 사용하려 했으나, transformer 사용이 좀 더 쉬운 [CLIP](https://github.com/openai/CLIP) 사용함.
- CLIP 으로 주어진 이미지들의 임베딩 벡터를 [FAISS](https://github.com/facebookresearch/faiss)에 두고, 입력받은 텍스트의 임베딩을 구해서 잠재 공간에서 가까운 순으로 출력하는 기능 작성.

# [1.1.0](https://github.com/maakcode/deep-image-search/releases/tag/1.1.0)

- CLIP이 다국어 기반으로 만들어지지 않아 입력한 키워드를 영어로 바꿔주는 소형 LLM을 도입했고, 소형 모델중에서 벤치마크가 좋게 나온다는 [IBM의 granite 4](https://huggingface.co/collections/ibm-granite/granite-40-language-modelsibm-granite/granite-4.0-micro) 선택
- ibm-granite/granite-4.0-h-350m 사용하려 했으나 번역 성능이 너무 나빠서(개구리 -> racoon으로 번역) 모델 크기 올리다보니 micro(3b)까지 올라감.

# [2.0.0](https://github.com/maakcode/deep-image-search/releases/tag/2.0.0)

- apple silicon에서 실행 성능이 낮아 기존 transformer를 cpu에서 mps로 개선하려했지만 어려워서 MLX 기반으로 변경함.

# [3.0.0](https://github.com/maakcode/deep-image-search/releases/tag/3.0.0)

- 다국어 지원을 위해 별도의 LLM으로 번역하는 대신 [sentence-transformers/clip-ViT-B-32-multilingual-v1](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1) 모델을 사용하도록 변경함.
- 원래는 [google/siglip2-so400m-patch16-naflex](https://huggingface.co/google/siglip2-so400m-patch16-naflex) 사용하려 했으나 이미지와 텍스트의 벡터 유사도가 제대로 계산하지 못하는 문제를 겪음([해당 구현](https://github.com/maakcode/deep-image-search/compare/2.0.0...2.0.0/siglip2))
- 모델 변경으로 인해 별도의 번역 과정이 필요 없어지면서 구조가 단순해지고 다국어 검색 성능이 개선됨.
