"""Embeds PR chunks using sentence-transformers and upserts them into Qdrant."""

import logging
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

from ingestion.chunker import Chunk
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


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
    ) -> int:
        """Embed and upsert new chunks; skip any already present in the vector store."""
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
            show_progress_bar=len(new_chunks) > batch_size,
            convert_to_numpy=True,
        )

        upsert_payloads = [
            {
                "id": chunk.id,
                "text": chunk.text,
                "vector": vec.tolist(),
                "metadata": chunk.metadata,
            }
            for chunk, vec in zip(new_chunks, vectors)
        ]
        vector_store.upsert(upsert_payloads)

        logger.info(
            "Embedded %d new chunks, skipped %d existing",
            len(new_chunks),
            len(existing_ids),
        )
        return len(new_chunks)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for search."""
        return self.model.encode(query, convert_to_numpy=True).tolist()
