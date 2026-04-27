"""Qdrant client wrapper providing upsert, search, and collection management."""

import hashlib
import logging
import os
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

load_dotenv()
logger = logging.getLogger(__name__)


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert an arbitrary string ID to a deterministic UUID-format string."""
    h = hashlib.md5(chunk_id.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


class VectorStore:
    """Thin wrapper around QdrantClient for PR chunk storage and retrieval."""

    # IMPORTANT: dimension must match the embedding model in models.py
    # BAAI/bge-base-en-v1.5 = 768, all-MiniLM-L6-v2 = 384
    def __init__(self, collection_name: str, vector_size: int = 768) -> None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s'", self.collection_name)

    def collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)

    def upsert(self, chunks: list[dict[str, Any]]) -> None:
        """Upsert a list of chunks, each with 'id', 'vector', and 'metadata' keys."""
        points = [
            PointStruct(
                id=_chunk_id_to_uuid(chunk["id"]),
                vector=chunk["vector"],
                payload={**chunk["metadata"], "text": chunk["text"], "chunk_id": chunk["id"]},
            )
            for chunk in chunks
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.debug("Upserted %d points into '%s'", len(points), self.collection_name)

    def get_existing_ids(self, chunk_ids: list[str]) -> set[str]:
        """Return the subset of chunk_ids already present in the collection."""
        if not chunk_ids:
            return set()
        uuid_to_original = {_chunk_id_to_uuid(cid): cid for cid in chunk_ids}
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=list(uuid_to_original.keys()),
            with_payload=False,
            with_vectors=False,
        )
        found_uuids = {r.id for r in results}
        return {uuid_to_original[uid] for uid in found_uuids if uid in uuid_to_original}

    def search(
        self,
        query_vector: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Search the collection and return payloads with scores."""
        qdrant_filter = self._build_filter(filters) if filters else None
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            {**r.payload, "score": r.score}
            for r in results.points
        ]

    def _build_filter(self, filters: dict[str, Any]) -> models.Filter:
        must: list[models.Condition] = []

        chunk_types = filters.get("chunk_types")
        if chunk_types:
            must.append(
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchAny(any=chunk_types),
                )
            )

        file_filters = filters.get("file_filters")
        if file_filters:
            must.append(
                models.FieldCondition(
                    key="filename",
                    match=models.MatchAny(any=file_filters),
                )
            )

        pr_number_lt = filters.get("pr_number_lt")
        if pr_number_lt is not None:
            must.append(
                models.FieldCondition(
                    key="pr_number",
                    range=models.Range(lt=pr_number_lt),
                )
            )

        return models.Filter(must=must) if must else models.Filter()

    def fetch_pr_titles(self, pr_numbers: list[int]) -> dict[int, str]:
        """Return {pr_number: title} by fetching metadata chunks for the given PRs."""
        if not pr_numbers:
            return {}
        scroll_filter = models.Filter(must=[
            models.FieldCondition(key="chunk_type", match=models.MatchValue(value="metadata")),
            models.FieldCondition(key="pr_number", match=models.MatchAny(any=pr_numbers)),
        ])
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=len(pr_numbers),
            with_payload=True,
            with_vectors=False,
        )
        result: dict[int, str] = {}
        for r in records:
            payload = r.payload or {}
            pr_num = payload.get("pr_number")
            if pr_num is None:
                continue
            title = (payload.get("title") or "").strip()
            if not title:
                text = payload.get("text", "")
                first_line = text.split("\n")[0]
                prefix = f"PR #{pr_num}: "
                if first_line.startswith(prefix):
                    title = first_line[len(prefix):].strip()
            if title:
                result[pr_num] = title
        return result

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        logger.info("Deleted Qdrant collection '%s'", self.collection_name)

    @staticmethod
    def list_collections() -> list[dict]:
        """Return info for all collections in Qdrant."""
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        client = QdrantClient(host=host, port=port)
        result = []
        for c in client.get_collections().collections:
            info = client.get_collection(c.name)
            count = getattr(info, "points_count", None) or getattr(info, "vectors_count", None) or 0
            result.append({
                "name": c.name,
                "vectors_count": count,
                "status": str(info.status),
            })
        return result
