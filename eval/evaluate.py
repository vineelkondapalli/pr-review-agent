"""Evaluation harness: retrieval recall, citation accuracy, and LLM-as-judge review relevance.

Usage:
  python eval/evaluate.py --repo owner/repo [--holdout 30] [--limit 200]
"""

import argparse
import logging
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv
from github import Auth, Github

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.critic import Critic
from agents.executor import Executor
from agents.planner import Planner
from agents.synthesizer import Synthesizer
from ingestion.chunker import chunk_pr
from ingestion.embedder import Embedder
from ingestion.github_fetcher import GitHubFetcher
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are evaluating an AI-generated code review.

Incoming PR diff:
{diff}

Generated review:
{review}

Rate the review's relevance and usefulness on a scale of 1-5:
1 = Completely off-topic or useless
2 = Somewhat related but mostly unhelpful
3 = Relevant but generic, no specific insight
4 = Specific and useful with some grounded observations
5 = Excellent, specific, well-cited, actionable

Return ONLY a single integer (1-5), nothing else."""


@dataclass
class EvalRecord:
    pr_number: int
    retrieval_recall: float
    citation_accurate: bool
    relevance_score: int


def _extract_pr_references(text: str) -> set[int]:
    return {int(m) for m in re.findall(r"#(\d+)", text)}


def _get_diff(pr_obj) -> str:
    parts = []
    for f in pr_obj.get_files():
        if f.patch:
            parts.append(f"--- a/{f.filename}\n+++ b/{f.filename}\n{f.patch}")
    return "\n\n".join(parts)


def _judge_relevance(client: anthropic.Anthropic, diff: str, review: str) -> int:
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=8,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(diff=diff[:3000], review=review[:3000]),
            }],
        )
        return int(resp.content[0].text.strip()[0])
    except Exception:
        return 0


class EvalRunner:
    def __init__(self, repo_str: str, holdout: int = 30, limit: int = 200) -> None:
        self.repo_str = repo_str
        self.holdout = holdout
        self.limit = limit
        self.collection = repo_str.replace("/", "_") + "_eval"

    def run(self, progress_callback: Callable[[str, int, int], None] | None = None) -> list[EvalRecord]:
        token = os.environ["GITHUB_TOKEN"]
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        fetcher = GitHubFetcher(token=token, repo_str=self.repo_str)
        all_prs = fetcher.fetch_prs(limit=self.limit)

        if len(all_prs) <= self.holdout:
            raise ValueError(f"Not enough PRs ({len(all_prs)}) for holdout of {self.holdout}")

        train_prs = all_prs[self.holdout:]
        holdout_prs = all_prs[:self.holdout]

        # Ingest training PRs
        vs = VectorStore(collection_name=self.collection)
        embedder = Embedder()
        reranker = Reranker()

        all_chunks = []
        for pr in train_prs:
            all_chunks.extend(chunk_pr(pr, self.repo_str))
        embedder.embed_chunks(all_chunks, vs)
        if progress_callback:
            progress_callback("ingest", 1, 1)

        planner = Planner(client)
        executor = Executor(vs, embedder, reranker)
        synthesizer = Synthesizer(client)
        critic = Critic(client)

        # Fetch holdout PRs via PyGithub for diff access
        gh = Github(auth=Auth.Token(token))
        gh_repo = gh.get_repo(self.repo_str)

        records: list[EvalRecord] = []
        train_pr_numbers = {pr.pr_number for pr in train_prs}
        done = 0

        for pr_data in holdout_prs:
            try:
                gh_pr = gh_repo.get_pull(pr_data.pr_number)
                diff = _get_diff(gh_pr)
                if not diff.strip():
                    continue

                # Ground truth: PR numbers referenced in review comments
                gt_refs: set[int] = set()
                for rc in pr_data.review_comments:
                    gt_refs |= _extract_pr_references(rc.body)
                for comment in pr_data.general_comments:
                    gt_refs |= _extract_pr_references(comment)
                gt_refs &= train_pr_numbers  # only count refs to training PRs

                # Run pipeline
                plan = planner.plan(diff)
                context_chunks = executor.execute(plan)
                synthesis = synthesizer.synthesize(diff, context_chunks)
                final = critic.verify(synthesis.raw_markdown, context_chunks)

                # Retrieval recall
                retrieved_prs = {c.get("pr_number") for c in context_chunks if c.get("pr_number")}
                if gt_refs:
                    recall = len(gt_refs & retrieved_prs) / len(gt_refs)
                else:
                    recall = 1.0  # no ground truth refs → vacuously perfect

                record = EvalRecord(
                    pr_number=pr_data.pr_number,
                    retrieval_recall=recall,
                    citation_accurate=final.verified,
                    relevance_score=_judge_relevance(client, diff, final.cleaned_review),
                )
                records.append(record)

            except Exception as exc:
                logger.warning("PR #%d eval failed: %s", pr_data.pr_number, exc)

            done += 1
            if progress_callback:
                progress_callback("eval", done, len(holdout_prs))

        return records


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate the PR review pipeline")
    parser.add_argument("--repo", required=True, help="GitHub repo (owner/repo)")
    parser.add_argument("--holdout", type=int, default=30, help="Number of holdout PRs")
    parser.add_argument("--limit", type=int, default=200, help="Total PRs to fetch")
    args = parser.parse_args()

    runner = EvalRunner(args.repo, holdout=args.holdout, limit=args.limit)
    runner.run()


if __name__ == "__main__":
    main()
