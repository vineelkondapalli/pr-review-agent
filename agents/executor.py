"""Agent 2: runs retrieval sub-queries, deduplicates, and reranks results."""

import logging
from typing import Any

from agents.planner import PlannerOutput
from ingestion.embedder import Embedder
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Executor:
    """Runs each planner query against Qdrant, deduplicates, and reranks the results."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        per_query_top_k: int = 40,
        final_top_k: int = 20,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.per_query_top_k = per_query_top_k
        self.final_top_k = final_top_k

    def execute(self, plan: PlannerOutput) -> list[dict[str, Any]]:
        """Run all sub-queries, merge, deduplicate, rerank, and return top chunks."""
        all_chunks: list[dict[str, Any]] = []

        filters: dict[str, Any] = {}
        if plan.chunk_types:
            filters["chunk_types"] = plan.chunk_types
        if plan.file_filters:
            filters["file_filters"] = plan.file_filters

        for query in plan.queries:
            try:
                query_vector = self.embedder.embed_query(query)
                results = self.vector_store.search(
                    query_vector=query_vector,
                    filters=filters or None,
                    top_k=self.per_query_top_k,
                )
                all_chunks.extend(results)
                logger.debug("Query '%s...' returned %d chunks", query[:50], len(results))
            except Exception as exc:
                logger.warning("Query failed: %s — %s", query[:50], exc)

        if not all_chunks:
            logger.warning("No chunks retrieved across %d queries", len(plan.queries))
            return []

        # Deduplicate across queries (keep highest score per chunk_id)
        seen: dict[str, dict[str, Any]] = {}
        for chunk in all_chunks:
            cid = chunk.get("chunk_id", "")
            if cid not in seen or chunk.get("score", 0) > seen[cid].get("score", 0):
                seen[cid] = chunk
        unique_chunks = list(seen.values())
        logger.info("Merged %d chunks → %d unique after dedup", len(all_chunks), len(unique_chunks))

        # Combine queries into one string for reranking
        combined_query = " ".join(plan.queries)
        reranked = self.reranker.rerank(combined_query, unique_chunks, top_k=self.final_top_k)
        logger.info("Returning %d reranked chunks", len(reranked))
        return reranked
