from sentence_transformers import SentenceTransformer
from typing import Union
import numpy as np

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

class Embedder:

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        print(f"[Embedder] Loading model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"[Embedder] Model ready.")

    def embed(self, texts: Union[str, list[str]]) -> np.ndarray:
        if isinstance(texts, str):
            return self.model.encode(texts, convert_to_numpy=True)
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed(query)