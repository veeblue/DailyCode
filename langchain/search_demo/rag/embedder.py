from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings


class RagEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def load_huggingface_embeddings(self, model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
        """加载HuggingFace嵌入模型"""
        if model_name:
            self.model_name = model_name

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.model_type = "huggingface"
            return self.embeddings

        except Exception as e:
            raise Exception(f"加载HuggingFace嵌入模型失败: {str(e)}")