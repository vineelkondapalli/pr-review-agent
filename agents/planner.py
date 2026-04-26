"""Agent 1: analyzes an incoming PR diff and produces structured retrieval sub-queries."""

import json
import logging
import re
from dataclasses import dataclass, field

import anthropic

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You analyze PR diffs and produce a structured retrieval plan.

Return ONLY valid JSON with this exact structure:
{
  "queries": ["query1", "query2"],
  "file_filters": ["path/to/file.py"],
  "chunk_types": ["diff", "review_comment", "metadata"],
  "recency_bias": false
}

Rules:
- queries: 1–5 specific semantic search queries targeting what this PR changes
- file_filters: list of filenames most relevant to the change (empty list if broad)
- chunk_types: which chunk types to retrieve ("diff", "review_comment", "metadata")
- recency_bias: true only if the PR touches release/version/changelog files
- Return ONLY the JSON object, no markdown fences, no explanation.
"""


@dataclass
class PlannerOutput:
    queries: list[str]
    file_filters: list[str]
    chunk_types: list[str]
    recency_bias: bool


def _truncate_diff(diff: str, max_chars: int = 8000) -> str:
    if len(diff) <= max_chars:
        return diff
    half = max_chars // 2
    return diff[:half] + "\n\n[... diff truncated ...]\n\n" + diff[-half:]


def _extract_json(text: str) -> dict:
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(text)


class Planner:
    """Analyzes a PR diff and generates targeted retrieval queries."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self.client = client

    def plan(self, pr_diff: str) -> PlannerOutput:
        truncated = _truncate_diff(pr_diff)
        last_error = ""

        for attempt in range(3):
            prompt = f"Analyze this PR diff and return the retrieval plan JSON.\n\n{truncated}"
            if last_error:
                prompt += f"\n\nPrevious response was invalid JSON: {last_error}\nReturn ONLY the JSON object."

            try:
                response = self.client.messages.create(
                    model=MODEL,
                    max_tokens=512,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text.strip()
                data = _extract_json(raw)

                queries = data.get("queries", [])[:5]
                if not queries:
                    raise ValueError("No queries returned")

                return PlannerOutput(
                    queries=queries,
                    file_filters=data.get("file_filters", []),
                    chunk_types=data.get("chunk_types", ["diff", "review_comment"]),
                    recency_bias=bool(data.get("recency_bias", False)),
                )
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = str(exc)
                logger.warning("Planner parse error (attempt %d): %s", attempt + 1, exc)
            except Exception as exc:
                logger.warning("Planner API error (attempt %d): %s", attempt + 1, exc)
                last_error = str(exc)

        # Fallback: use the diff itself as a single broad query
        logger.error("Planner failed after 3 attempts, using fallback query")
        return PlannerOutput(
            queries=[truncated[:200]],
            file_filters=[],
            chunk_types=["diff", "review_comment"],
            recency_bias=False,
        )
