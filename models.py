"""Global model registry — loads ML models in a background thread at startup."""

import logging
import os
import threading
import warnings
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Set HF_HOME before any HF imports so both SentenceTransformer and CrossEncoder
# pick up the same cache directory.
_cache = os.getenv("REVUE_MODEL_CACHE")
if _cache:
    os.environ.setdefault("HF_HOME", _cache)

# Silence HF/transformers noise before any imports.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
# Force offline mode so cached models load from disk without HF Hub network calls.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Force single-threaded OpenMP/MKL to prevent deadlocks on Windows when PyTorch's
# internal thread pool collides with asyncio's IocpProactor thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Suppress the unauthenticated-request warning regardless of warning category.
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*HF Hub.*")

# Root-level logging filter — catches the message no matter which logger emits it.
_HF_NOISE = ("unauthenticated", "HF_TOKEN", "higher rate limits")


class _SuppressHFWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(token in msg for token in _HF_NOISE)


logging.getLogger().addFilter(_SuppressHFWarnings())


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

_embedder: Any = None
_reranker: Any = None
_ready = threading.Event()
_load_error: BaseException | None = None


_load_lock = threading.Lock()
_load_started = False


def _load() -> None:
    global _embedder, _reranker, _load_error
    try:
        import tqdm as _tqdm
        _orig_init = _tqdm.tqdm.__init__

        def _silent_init(self, *a, **kw):
            kw["disable"] = True
            _orig_init(self, *a, **kw)

        _tqdm.tqdm.__init__ = _silent_init
    except ImportError:
        pass

    try:
        logger.info("_load: importing sentence_transformers")
        from sentence_transformers import CrossEncoder, SentenceTransformer
        logger.info("_load: loading SentenceTransformer")
        _embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=_device)
        logger.info("_load: loading CrossEncoder")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=_device)
        logger.info("_load: done")
    except BaseException as exc:
        _load_error = exc
        logger.error("Model loading failed: %s", exc, exc_info=True)
    finally:
        _ready.set()


def load_async() -> None:
    """Start loading models in a background daemon thread. Idempotent."""
    global _load_started
    with _load_lock:
        if _load_started:
            return
        _load_started = True
    threading.Thread(target=_load, daemon=True, name="revue-model-loader").start()


def load_sync() -> None:
    """Load models on the calling thread. Idempotent. Raises on failure."""
    global _load_started
    with _load_lock:
        already_started = _load_started
        _load_started = True
    if already_started:
        wait()
        return
    _load()
    if _load_error is not None:
        raise RuntimeError(f"Model loading failed: {_load_error}") from _load_error


def is_ready() -> bool:
    return _ready.is_set()


def wait() -> None:
    """Block until models are loaded. Raises RuntimeError on failure."""
    _ready.wait()
    if _load_error is not None:
        raise RuntimeError(f"Model loading failed: {_load_error}") from _load_error


class _Proxy:
    """Blocks on first attribute access until the background load completes."""

    def __init__(self, ref: Callable[[], Any]) -> None:
        object.__setattr__(self, "_ref", ref)

    def __getattr__(self, name: str) -> Any:
        wait()
        return getattr(object.__getattribute__(self, "_ref")(), name)


EMBEDDER: Any = _Proxy(lambda: _embedder)
RERANKER: Any = _Proxy(lambda: _reranker)
