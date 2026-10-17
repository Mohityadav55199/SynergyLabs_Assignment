import os
from typing import List, Optional
from fastembed import TextEmbedding
from src.config import settings

class Embedder:
    """
    Local embedding engine powered by FastEmbed (ONNX runtime).
    Generates high quality, 384-dimensional embeddings locally with zero API cost.
    Records model metadata and dimensionality as per assignment requirements.
    """
    def __init__(self, model_name: Optional[str] = None):
        # Disable huggingface hub symlink warning on Windows
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        self.model_name = model_name or settings.embedding_model
        self.client = TextEmbedding(self.model_name)
        self.embedding_dim = settings.embedding_dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings into normalized float vectors.
        """
        if not texts:
            return []
        embeddings = list(self.client.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single search query string.
        """
        embeddings = list(self.client.embed([text]))
        return embeddings[0].tolist()

    @property
    def metadata(self) -> dict:
        return {
            "model_name": self.model_name,
            "dimensionality": self.embedding_dim,
            "provider": "fastembed-local-onnx"
        }

# Global embedder singleton
_embedder_instance = None

def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
