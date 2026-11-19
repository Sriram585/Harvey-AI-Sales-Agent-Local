"""Local text embeddings using offline sentence-transformers."""

import os
import math
from pathlib import Path

# Enforce strictly offline HuggingFace Hub operation
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

LOCAL_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "local_models" / "all-MiniLM-L6-v2"

_model = None


def _get_model():
    """Lazy load the sentence transformer model strictly from local disk."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        resolved_path = LOCAL_MODEL_PATH.resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Local embeddings model not found at {resolved_path}. "
                "Run 'python scripts/setup_models.py' to cache the model first."
            )
        _model = SentenceTransformer(str(resolved_path), local_files_only=True)
    return _model


def embed_text(text: str) -> list[float]:
    """Generate dense vector embedding for input text using local model."""
    model = _get_model()
    # Normalize embeddings for cosine similarity
    vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return [float(x) for x in vec.tolist()]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))
