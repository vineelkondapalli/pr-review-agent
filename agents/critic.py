"""Agent 4: verifies that every citation in the review traces to a real retrieved chunk."""

import logging
import re
from dataclasses import dataclass
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"

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


def _find_hallucinated(citations: set[str], valid_pr_numbers: set[int], valid_filenames: set[str]) -> list[str]:
    hallucinated = []
    for cite in sorted(citations):
        pr_match = re.match(r"\[PR #(\d+)\]$", cite)
        if pr_match:
            if int(pr_match.group(1)) not in valid_pr_numbers:
                hallucinated.append(cite)
            continue
        ref_match = re.match(r"\[ref:\s*([^\]]+)\]$", cite)
        if ref_match:
            if ref_match.group(1).strip() not in valid_filenames:
                hallucinated.append(cite)
    return hallucinated


@dataclass
class CriticResult:
    verified: bool
    hallucinated_refs: list[str]
    cleaned_review: str


class Critic:
    """Verifies and cleans hallucinated citations from synthesizer output."""

    def __init__(self, client: anthropic.Anthropic, model: str = MODEL) -> None:
        self.client = client
        self.model = model

    def verify(self, review_markdown: str, context_chunks: list[dict[str, Any]]) -> CriticResult:
        if not context_chunks:
            return CriticResult(verified=True, hallucinated_refs=[], cleaned_review=review_markdown)

        citations = _extract_citations(review_markdown)
        if not citations:
            return CriticResult(verified=True, hallucinated_refs=[], cleaned_review=review_markdown)

        valid_pr_numbers, valid_filenames = _build_valid_refs(context_chunks)
        hallucinated = _find_hallucinated(citations, valid_pr_numbers, valid_filenames)

        if not hallucinated:
            return CriticResult(verified=True, hallucinated_refs=[], cleaned_review=review_markdown)

        logger.warning("Critic found hallucinated refs: %s", hallucinated)
        cleaned = self._clean_review(review_markdown, hallucinated)
        return CriticResult(verified=False, hallucinated_refs=hallucinated, cleaned_review=cleaned)

    def _clean_review(self, review_markdown: str, hallucinated: list[str]) -> str:
        refs_str = ", ".join(hallucinated)
        try:
            response = self.client.messages.create(
                model=self.model,
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
