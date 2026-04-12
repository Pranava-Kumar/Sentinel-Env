"""Production Text Embedder for Sentinel Environment.

Provides real 384-dim text embeddings using sentence-transformers.
Fixes critical missing component - all training scripts reference this
but the file didn't exist, causing training to use random noise.
"""

import numpy as np
import structlog

logger = structlog.get_logger()


class TextEmbedder:
    """Production text embedding using sentence-transformers.

    Encodes text to 384-dimensional embeddings using pre-trained models.
    Falls back to random embeddings if model loading fails (with warning).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", fallback_dim: int = 384):
        """Initialize text embedder.

        Args:
            model_name: Name of sentence-transformers model to use.
            fallback_dim: Dimension of random fallback embeddings.
        """
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load sentence-transformers model with fallback."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformers model", model=self.model_name)
            model = SentenceTransformer(self.model_name)
            self.model = model
            # Use new method name if available, fallback to old
            get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
            self.embedding_dim: int = get_dim() if get_dim else self.fallback_dim
            self.embedding_dim = self.embedding_dim or self.fallback_dim
            logger.info("Text embedder loaded successfully", dim=self.embedding_dim)
        except Exception as e:
            logger.warning(
                "Failed to load sentence-transformers, using random fallback", error=str(e), dim=self.fallback_dim
            )
            self.model = None
            self.embedding_dim = self.fallback_dim

    def encode(self, text: str) -> np.ndarray:
        """Encode single text to embedding vector.

        Args:
            text: Input text to encode.

        Returns:
            Numpy array of shape (embedding_dim,) with L2-normalized embeddings.
        """
        if self.model is not None:
            embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding
        else:
            # Fallback: random embeddings (training won't learn real patterns)
            logger.warning("Using random embedding fallback - training cannot learn patterns")
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding

    def encode_batch(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode multiple texts to embedding vectors.

        Args:
            texts: List of input texts to encode.
            show_progress: Whether to show progress bar.

        Returns:
            Numpy array of shape (len(texts), embedding_dim) with L2-normalized embeddings.
        """
        if self.model is not None:
            embeddings = self.model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=show_progress
            )
            return embeddings
        else:
            # Fallback: random embeddings
            logger.warning("Using random batch embeddings fallback")
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            # L2 normalize each row
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Avoid division by zero
            embeddings = embeddings / norms
            return embeddings

    def encode_prompt(self, prompt: str, response: str | None = None) -> np.ndarray:
        """Encode prompt optionally combined with response.

        Args:
            prompt: Input prompt text.
            response: Optional response text to append.

        Returns:
            Numpy array of shape (embedding_dim,) with combined embedding.
        """
        text = f"{prompt}\n{response}" if response else prompt
        return self.encode(text)


# Singleton instance for convenience
_default_embedder: TextEmbedder | None = None


def get_embedder() -> TextEmbedder:
    """Get or create default text embedder singleton.

    Returns:
        TextEmbedder instance.
    """
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = TextEmbedder()
    return _default_embedder


def encode_text(text: str) -> np.ndarray:
    """Convenience function to encode text using default embedder.

    Args:
        text: Input text to encode.

    Returns:
        Numpy array of shape (384,) with embedding.
    """
    return get_embedder().encode(text)


def encode_batch(texts: list[str]) -> np.ndarray:
    """Convenience function to encode batch of texts.

    Args:
        texts: List of input texts.

    Returns:
        Numpy array of shape (len(texts), 384) with embeddings.
    """
    return get_embedder().encode_batch(texts)
