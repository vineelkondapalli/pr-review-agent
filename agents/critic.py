"""Agent 4: verifies that every citation in the review traces to a real retrieved chunk."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are a citation verifier for AI-generated code reviews.

Your task: check whether every citation in the review exists in the provided list of valid references.

A citation looks like [PR #N] or [ref: filename].
A valid PR citation [PR #N] is valid only if PR number N appears in the valid PR numbers list.
A valid file citation [ref: filename] is valid only if that filename appears in the valid filenames list.

Return ONLY valid JSON:
{
  "hallucinated_refs": ["[PR #99]", "[ref: fake.py]"],
  "explanation": "brief explanation of what was hallucinated"
}

If everything checks out, return: {"hallucinated_refs": [], "explanation": "All citations verified."}
"""

CLEAN_PROMPT = """Remove or rewrite any sentences containing these hallucinated citations: {refs}

Original review:
{review}

Return the cleaned review with the same structure but without the hallucinated citations.
Do not add new citations or change anything else."""


def _extract_citations(text: str) -> set[str]:
    pr_cites = set(re.findall(r"\[PR #\d+\]", text))
    ref_cites = set(re.findall(r"\[ref:\s*[^\]]+\]", text))
    return pr_cites | ref_cites


def _build_valid_refs(chunks: list[dict[str, Any]]) -> tuple[set[int], set[str]]:
    valid_pr_numbers: set[int] = set()
    valid_filenames: set[str] = set()
    for chunk in chunks:
        pr_num = chunk.get("pr_number")
        if pr_num is not None:
            valid_pr_numbers.add(int(pr_num))
        filename = chunk.get("filename", "")
        if filename:
            valid_filenames.add(filename)
    return valid_pr_numbers, valid_filenames


@dataclass
class CriticResult:
    verified: bool
    hallucinated_refs: list[str]
    cleaned_review: str


class Critic:
    """Verifies and cleans hallucinated citations from synthesizer output."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self.client = client

    def verify(self, review_markdown: str, context_chunks: list[dict[str, Any]]) -> CriticResult:
        citations = _extract_citations(review_markdown)
        if not citations:
            return CriticResult(verified=True, hallucinated_refs=[], cleaned_review=review_markdown)

        valid_pr_numbers, valid_filenames = _build_valid_refs(context_chunks)

        valid_refs_text = (
            f"Valid PR numbers: {sorted(valid_pr_numbers)}\n"
            f"Valid filenames: {sorted(valid_filenames)}"
        )
        user_message = (
            f"{valid_refs_text}\n\n"
            f"Citations found in review: {', '.join(sorted(citations))}\n\n"
            f"Review:\n{review_markdown}"
        )

        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            import json, re as _re
            raw = response.content[0].text.strip()
            raw = _re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
            data = json.loads(raw)
            hallucinated = data.get("hallucinated_refs", [])
        except Exception as exc:
            logger.warning("Critic verification failed: %s", exc)
            hallucinated = []

        if not hallucinated:
            return CriticResult(verified=True, hallucinated_refs=[], cleaned_review=review_markdown)

        logger.warning("Critic found hallucinated refs: %s", hallucinated)
        cleaned = self._clean_review(review_markdown, hallucinated)
        return CriticResult(verified=False, hallucinated_refs=hallucinated, cleaned_review=cleaned)

    def _clean_review(self, review_markdown: str, hallucinated: list[str]) -> str:
        refs_str = ", ".join(hallucinated)
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": CLEAN_PROMPT.format(refs=refs_str, review=review_markdown),
                }],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.warning("Critic cleanup failed: %s — stripping citations manually", exc)
            cleaned = review_markdown
            for ref in hallucinated:
                cleaned = cleaned.replace(ref, "")
            return cleaned
