"""Multi-turn RAG chatbot agent with streaming responses."""

from __future__ import annotations

import logging
from typing import Any, Generator

import anthropic

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
MAX_HISTORY_TURNS = 10  # 10 turns = 20 messages (user + assistant)

SYSTEM_PROMPT = """You are an expert assistant for analyzing a GitHub repository's PR history.
Answer questions by citing specific PR numbers and filenames from the provided context.
Use [PR #N] citations when referencing specific pull requests.
Be concise and ground every claim in the provided context.
If the context does not contain enough information to answer, say so clearly."""


def _format_chunks(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        pr_num = chunk.get("pr_number", "?")
        chunk_type = chunk.get("chunk_type", "?")
        filename = chunk.get("filename") or ""
        label = f"[Chunk {i + 1} | PR #{pr_num} | {chunk_type}" + (f" | {filename}" if filename else "") + "]"
        parts.append(f"{label}\n{chunk.get('text', '')}")
    return "\n\n---\n\n".join(parts)


class ChatAgent:
    """Multi-turn RAG chatbot with conversation history capped at MAX_HISTORY_TURNS."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        vector_store: Any,
        embedder: Any,
        reranker: Any,
        top_k: int = 15,
    ) -> None:
        self.client = client
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.top_k = top_k
        self.history: list[dict[str, str]] = []

    def reset(self) -> None:
        self.history.clear()

    def _retrieve(self, question: str) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed_query(question)
        results = self.vector_store.search(
            query_vector=query_vector,
            filters=None,
            top_k=self.top_k * 3,
        )
        return self.reranker.rerank(question, results, top_k=self.top_k)

    def stream_response(
        self, question: str
    ) -> Generator[tuple[str, list[dict[str, Any]] | None], None, None]:
        """Stream a response, yielding (delta, None) per chunk and ("", chunks) as final sentinel.

        Usage:
            buffer = ""
            for delta, chunks in agent.stream_response(question):
                if chunks is not None:
                    source_chunks = chunks
                else:
                    buffer += delta
        """
        source_chunks = self._retrieve(question)
        context_text = _format_chunks(source_chunks)

        user_content = (
            f"Context from repository PR history:\n\n{context_text}"
            f"\n\n---\n\nQuestion: {question}"
        )

        # Cap history at MAX_HISTORY_TURNS pairs
        capped = self.history[-(MAX_HISTORY_TURNS * 2):]
        messages = capped + [{"role": "user", "content": user_content}]

        full_response = ""
        with self.client.messages.stream(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for text_delta in stream.text_stream:
                full_response += text_delta
                yield text_delta, None

        # Store the raw question (not context-injected) in history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": full_response})

        yield "", source_chunks
