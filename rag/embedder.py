"""
rag/embedder.py

Wraps sentence-transformers to convert text into vector embeddings.
The model is loaded once and reused across all calls.

Adding new classes: no changes needed here — the embedder is class-agnostic.
"""

from sentence_transformers import SentenceTransformer
from typing import Union
import numpy as np


_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Loads a sentence-transformer model and exposes a single embed() method.
    Works for both single strings and lists of strings.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        """
        Args:
            model_name: Any sentence-transformers compatible model name.
                        Defaults to all-MiniLM-L6-v2 (fast, CPU-friendly, 384-dim).
        """
        print(f"[Embedder] Loading model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"[Embedder] Model ready.")

    def embed(self, texts: Union[str, list[str]]) -> np.ndarray:
        """
        Embed one or more strings into dense vectors.

        Args:
            texts: A single string or a list of strings.

        Returns:
            numpy array of shape (n, embedding_dim) for lists,
            or (embedding_dim,) for a single string.
        """
        if isinstance(texts, str):
            return self.model.encode(texts, convert_to_numpy=True)
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_query(self, query: str) -> np.ndarray:
        """Convenience wrapper for single query embedding."""
        return self.embed(query)