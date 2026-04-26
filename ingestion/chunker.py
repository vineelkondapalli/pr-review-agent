"""Splits a PRData object into metadata, per-file diff, and review comment chunks."""

import hashlib
from dataclasses import dataclass
from typing import Any

from ingestion.github_fetcher import PRData


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


def _make_chunk_id(repo: str, pr_number: int, chunk_type: str, filename: str = "") -> str:
    raw = f"{repo}:{pr_number}:{chunk_type}:{filename}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def chunk_pr(pr: PRData, repo: str) -> list[Chunk]:
    """Produce metadata, diff, and review_comment chunks for a single PR."""
    chunks: list[Chunk] = []
    outcome = "merged" if pr.merged else "closed"
    base_meta = {
        "pr_number": pr.pr_number,
        "author": pr.author,
        "date": pr.created_at,
        "outcome": outcome,
        "repo": repo,
    }

    # Metadata chunk
    labels_str = ", ".join(pr.labels) if pr.labels else "none"
    metadata_text = (
        f"PR #{pr.pr_number}: {pr.title}\n"
        f"{pr.description}\n"
        f"Labels: {labels_str}\n"
        f"Outcome: {outcome}"
    )
    chunks.append(Chunk(
        id=_make_chunk_id(repo, pr.pr_number, "metadata"),
        text=metadata_text.strip(),
        metadata={**base_meta, "chunk_type": "metadata", "filename": ""},
    ))

    # Per-file diff chunks
    for file_data in pr.files:
        if not file_data.patch:
            continue
        diff_text = f"File: {file_data.filename}\n{file_data.patch}"
        chunks.append(Chunk(
            id=_make_chunk_id(repo, pr.pr_number, "diff", file_data.filename),
            text=diff_text.strip(),
            metadata={**base_meta, "chunk_type": "diff", "filename": file_data.filename},
        ))

    # Review comment chunks
    for rc in pr.review_comments:
        comment_text = f"Review on {rc.path}:{rc.line or '?'}\n{rc.body}"
        chunks.append(Chunk(
            id=_make_chunk_id(repo, pr.pr_number, "review_comment", f"{rc.path}:{rc.line}"),
            text=comment_text.strip(),
            metadata={**base_meta, "chunk_type": "review_comment", "filename": rc.path},
        ))

    return chunks
