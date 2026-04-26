"""Global model registry — loads all ML models once at startup."""

import logging

from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)


def _best_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


_device = _best_device()
logger.info("Using device: %s", _device)

logger.info("Loading embedding model...")
EMBEDDER = SentenceTransformer("BAAI/bge-base-en-v1.5", device=_device)

logger.info("Loading reranker model...")
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=_device)

logger.info("All models loaded.")
