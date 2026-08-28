"""
embeddings.py - a single, lazily-loaded local SentenceTransformers encoder.

Kept in its own module so that rag_core.py, llm.py and the guardrails can all
share one loaded model instead of paying the load cost several times.
"""

from functools import lru_cache

import numpy as np

from config import EMBEDDING_MODEL

_model = None


def get_model():
    """Load all-MiniLM-L6-v2 once per process."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(texts, normalize: bool = True) -> np.ndarray:
    """Encode a list of strings into a (n, 384) float32 matrix."""
    if isinstance(texts, str):
        texts = [texts]
    vecs = get_model().encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)


@lru_cache(maxsize=4096)
def embed_one(text: str) -> tuple:
    """Cached single-string embedding, returned as a tuple so it is hashable."""
    return tuple(float(x) for x in embed([text])[0])


def cosine(a, b) -> float:
    """Cosine similarity between two vectors (safe on zero vectors)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
