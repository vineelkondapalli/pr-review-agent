"""Cross-encoder reranker for scoring and reordering retrieved chunks."""

import logging
from typing import Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """Reranks retrieved chunks using a cross-encoder relevance model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        logger.info("Loading reranker model: %s", model_name)
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Deduplicate by chunk_id, score each with the cross-encoder, return top_k."""
        if not chunks:
            return []

        # Deduplicate: keep highest vector-search score per chunk_id
        seen: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            cid = chunk.get("chunk_id", chunk.get("id", ""))
            if cid not in seen or chunk.get("score", 0) > seen[cid].get("score", 0):
                seen[cid] = chunk
        unique_chunks = list(seen.values())

        pairs = [(query, c.get("text", "")) for c in unique_chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(unique_chunks, scores):
            chunk["rerank_score"] = float(score)

        ranked = sorted(unique_chunks, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
