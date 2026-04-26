"""Agent 3: synthesizes a grounded code review from a PR diff and retrieved context."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are an expert code reviewer with deep knowledge of software engineering best practices.

Your task: given an incoming PR diff and a set of retrieved context chunks from past PRs in the same repository, write a thorough, grounded code review.

Rules:
- Every specific claim, suggestion, or pattern observation MUST include a citation.
- Use [PR #N] to cite a past PR number, or [ref: filename] to cite a file from context.
- If you have no relevant context for a claim, do not make the claim.
- Structure your review with these exact markdown headers:
  ## Summary
  ## Per-File Comments
  ## Patterns Detected
  ## Verdict

Verdict must be one of: "Approve", "Request Changes", or "Nitpicks Only".
"""


@dataclass
class FileComment:
    filename: str
    comment: str


@dataclass
class Pattern:
    description: str


@dataclass
class SynthesisResult:
    summary: str
    file_comments: list[FileComment]
    patterns: list[Pattern]
    verdict: str
    raw_markdown: str


def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        pr_num = chunk.get("pr_number", "?")
        chunk_type = chunk.get("chunk_type", "?")
        filename = chunk.get("filename", "")
        label = f"[Chunk {i+1} | PR #{pr_num} | {chunk_type} | {filename}]"
        parts.append(f"{label}\n{chunk.get('text', '')}")
    return "\n\n---\n\n".join(parts)


def _parse_section(markdown: str, header: str) -> str:
    pattern = rf"##\s+{re.escape(header)}\s*\n(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _parse_result(markdown: str) -> SynthesisResult:
    summary = _parse_section(markdown, "Summary")
    per_file_raw = _parse_section(markdown, "Per-File Comments")
    patterns_raw = _parse_section(markdown, "Patterns Detected")
    verdict_raw = _parse_section(markdown, "Verdict")

    # Extract per-file comments (lines starting with `**filename**` or `### filename`)
    file_comments: list[FileComment] = []
    current_file = ""
    for line in per_file_raw.splitlines():
        file_match = re.match(r"###?\s*`?([^\`]+)`?\s*", line)
        if file_match:
            current_file = file_match.group(1).strip()
        elif line.strip() and current_file:
            file_comments.append(FileComment(filename=current_file, comment=line.strip()))

    # Extract bullet patterns
    patterns = [
        Pattern(description=line.lstrip("-* ").strip())
        for line in patterns_raw.splitlines()
        if line.strip().startswith(("-", "*", "•"))
    ]

    # Normalize verdict
    verdict = "Request Changes"
    verdict_lower = verdict_raw.lower()
    if "approve" in verdict_lower:
        verdict = "Approve"
    elif "nitpick" in verdict_lower:
        verdict = "Nitpicks Only"

    return SynthesisResult(
        summary=summary,
        file_comments=file_comments,
        patterns=patterns,
        verdict=verdict,
        raw_markdown=markdown,
    )


class Synthesizer:
    """Writes a grounded PR review citing retrieved context chunks."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self.client = client

    def synthesize(self, pr_diff: str, context_chunks: list[dict[str, Any]]) -> SynthesisResult:
        context_text = _format_context(context_chunks)
        user_message = (
            f"## Incoming PR Diff\n\n```diff\n{pr_diff}\n```\n\n"
            f"## Retrieved Context from Past PRs\n\n{context_text}"
        )

        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                raw = response.content[0].text
                if raw.strip():
                    return _parse_result(raw)
                logger.warning("Empty response from synthesizer (attempt %d)", attempt + 1)
            except Exception as exc:
                logger.warning("Synthesizer error (attempt %d): %s", attempt + 1, exc)

        # Return empty result if all attempts fail
        return SynthesisResult(
            summary="Review generation failed.",
            file_comments=[],
            patterns=[],
            verdict="Request Changes",
            raw_markdown="",
        )
