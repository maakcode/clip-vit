import os
from typing import Tuple
import streamlit as st
import faiss
from PIL import Image, ImageFile
from mlx_clip import CLIPModel, CLIPTokenizer, CLIPImageProcessor, convert_clip
from mlx_lm.generate import generate as mlx_generate
from mlx_lm.convert import convert as mlx_convert
from mlx_lm.utils import load as mlx_load

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class ClipImageSearch:
    def __init__(self):
        model_name = "openai/clip-vit-large-patch14"
        model_path = f"./models/{model_name}"

        self.index = None
        self.image_names = []
        self.image_objects = []

        st.write(f"🚀 CLIP 모델 로딩 중... ({model_name})")
        self.model, self.tokenizer, self.img_processor = self._load(
            model_name, model_path
        )
        st.success("✅ CLIP 모델 로드 완료!")

    def _load(
        self, model_name: str, model_path: str
    ) -> Tuple[CLIPModel, CLIPTokenizer, CLIPImageProcessor]:
        if not os.path.isdir(model_path):
            convert_clip(model_name, model_path)

        model = CLIPModel.from_pretrained(model_path)
        tokenizer = CLIPTokenizer.from_pretrained(model_path)
        img_processor = CLIPImageProcessor.from_pretrained(model_path)
        return model, tokenizer, img_processor

    def _get_image_embeddings(self, images: list[ImageFile.ImageFile]):
        """여러 이미지의 임베딩을 계산합니다."""
        rgb_images = [img.convert("RGB") for img in images]
        inputs = {
            "pixel_values": self.img_processor(rgb_images),
        }
        return self.model(**inputs).image_embeds

    def _get_text_embeddings(self, texts: list[str]):
        """여러 텍스트의 임베딩을 계산합니다."""
        inputs = {
            "input_ids": self.tokenizer(texts),
        }
        return self.model(**inputs).text_embeds

    def build_index(self, uploaded_files):
        """업로드된 이미지로 FAISS 인덱스 생성"""
        self.image_names = [f.name for f in uploaded_files]
        self.image_objects = [Image.open(f) for f in uploaded_files]

        embeddings = self._get_image_embeddings(self.image_objects)

        # FAISS index 생성
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        st.success(f"✅ {len(self.image_names)}개의 이미지가 인덱스에 추가되었습니다.")

    def search(self, query: str, top_k: int = 3):
        """텍스트 쿼리로 유사 이미지 검색"""
        if self.index is None:
            st.error("❌ 먼저 이미지 인덱스를 생성하세요.")
            return []

        text_emb = self._get_text_embeddings([query]).reshape(1, -1)
        D, I = self.index.search(text_emb, top_k)
        results = [
            (self.image_objects[i], self.image_names[i], float(D[0][idx]))
            for idx, i in enumerate(I[0])
        ]
        return results


class LLMTranslator:
    def __init__(self):
        model_name = "ibm-granite/granite-4.0-h-micro"
        model_path = f"./models/{model_name}"

        st.write(f"🚀 언어 모델 로딩 중... ({model_name})")
        self._convert(model_name, model_path)
        self.model, self.tokenizer, *rest = mlx_load(model_path)
        st.success("✅ 언어 모델 로드 완료!")

    def _convert(self, model_name: str, model_path: str):
        if os.path.isdir(model_path):
            return

        mlx_convert(model_name, quantize=True, mlx_path=model_path)

    def translate(self, text: str) -> str:
        chat = [
            {
                "role": "system",
                "content": "You are a translator. Translate the given text into English only, without any additional comments or responses.",
            },
            {
                "role": "user",
                "content": text,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        text = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
        )
        return text


def main():
    # ============================================
    # Streamlit UI
    # ============================================
    st.set_page_config(page_title="Image Search", layout="centered")
    st.title("🔍 이미지 검색")
    st.caption("이미지를 업로드하고 텍스트로 유사한 이미지를 검색합니다.")

    # 캐싱된 모델 사용
    @st.cache_resource
    def load_app():
        return ClipImageSearch()

    @st.cache_resource
    def load_translator():
        return LLMTranslator()

    app = load_app()
    translator = load_translator()

    # 파일 업로드
    uploaded_files = st.file_uploader(
        "이미지를 업로드하세요",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    # 인덱스 생성
    if uploaded_files:
        st.write("📦 업로드한 이미지:")
        cols = st.columns(5)
        for i, file in enumerate(uploaded_files):
            img = Image.open(file)
            cols[i % 5].image(img, caption=file.name, width="stretch")

        if st.button("🧠 이미지 인덱스 생성"):
            app.build_index(uploaded_files)

    # 검색
    if app.index is not None:
        query = st.text_input(
            "검색할 문장을 입력하세요 (예: '파란 눈을 가진 귀여운 고양이')"
        )
        top_k = st.slider("검색할 상위 이미지 개수", 1, 10, 3)

        if st.button("🔎 검색 실행"):
            translated_query = translator.translate(query)
            st.write(f"{query} {translated_query}")
            results = app.search(translated_query, top_k)
            if results:
                st.write("### 검색 결과:")
                cols = st.columns(top_k)

                for idx, (img, name, score) in enumerate(results):
                    cols[idx].image(
                        img,
                        caption=f"{name}\n유사도: {score:.3f}",
                        use_container_width=True,
                    )


if __name__ == "__main__":
    main()
