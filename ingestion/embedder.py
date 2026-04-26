"""Embeds PR chunks using sentence-transformers and upserts them into Qdrant."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer

from ingestion.chunker import Chunk
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_UPSERT_WORKERS = 4


class Embedder:
    """Embeds text chunks and upserts them into a VectorStore, skipping existing IDs."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_chunks(
        self,
        chunks: list[Chunk],
        vector_store: VectorStore,
        batch_size: int = 64,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Embed chunks in batches of `batch_size` and upsert batches to Qdrant in parallel."""
        if not chunks:
            return 0

        existing_ids = vector_store.get_existing_ids([c.id for c in chunks])
        new_chunks = [c for c in chunks if c.id not in existing_ids]

        if not new_chunks:
            logger.info("All %d chunks already in store, nothing to upsert", len(chunks))
            return 0

        texts = [c.text for c in new_chunks]
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Build batched upsert payloads
        batches: list[list[dict]] = []
        for batch_start in range(0, len(new_chunks), batch_size):
            batch_chunks = new_chunks[batch_start : batch_start + batch_size]
            batch_vecs = vectors[batch_start : batch_start + batch_size]
            batches.append([
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "vector": vec.tolist(),
                    "metadata": chunk.metadata,
                }
                for chunk, vec in zip(batch_chunks, batch_vecs)
            ])

        # Parallelize upserts across batches
        errors = 0
        done = 0
        with ThreadPoolExecutor(max_workers=_UPSERT_WORKERS) as pool:
            futures = {pool.submit(vector_store.upsert, batch): i for i, batch in enumerate(batches)}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors += 1
                    logger.error("Upsert batch %d failed: %s", futures[future], exc)
                done += 1
                if progress_callback is not None:
                    progress_callback(done, len(batches))

        if errors:
            logger.warning("%d upsert batch(es) failed out of %d", errors, len(batches))

        logger.info(
            "Embedded %d new chunks in %d batches, skipped %d existing",
            len(new_chunks),
            len(batches),
            len(existing_ids),
        )
        return len(new_chunks)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for search."""
        return self.model.encode(query, convert_to_numpy=True).tolist()
