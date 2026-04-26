"""Fetches PR history from a GitHub repository using the PyGithub API.

PRs are fetched concurrently (ThreadPoolExecutor, max 10 workers). Already-fetched
PRs are cached as JSON under cache/<repo_owner>_<repo_name>/ so subsequent runs
skip the network entirely for cached PRs.
"""

import dataclasses
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from github import Github, RateLimitExceededException
from github.PullRequest import PullRequest
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_MAX_WORKERS = 10
_CACHE_DIR = Path("cache")


@dataclass
class FileData:
    filename: str
    patch: Optional[str]


@dataclass
class ReviewComment:
    body: str
    path: str
    line: Optional[int]
    author: str


@dataclass
class PRData:
    pr_number: int
    title: str
    description: str
    labels: list[str]
    state: str
    merged: bool
    author: str
    created_at: str
    merged_at: Optional[str]
    files: list[FileData]
    review_comments: list[ReviewComment]
    general_comments: list[str]


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(cache_dir: Path, pr_number: int) -> Path:
    return cache_dir / f"{pr_number}.json"


def _to_dict(pr: PRData) -> dict:
    return dataclasses.asdict(pr)


def _from_dict(data: dict) -> PRData:
    return PRData(
        pr_number=data["pr_number"],
        title=data["title"],
        description=data["description"],
        labels=data["labels"],
        state=data["state"],
        merged=data["merged"],
        author=data["author"],
        created_at=data["created_at"],
        merged_at=data["merged_at"],
        files=[FileData(**f) for f in data["files"]],
        review_comments=[ReviewComment(**rc) for rc in data["review_comments"]],
        general_comments=data["general_comments"],
    )


def _save_cache(path: Path, pr: PRData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_dict(pr), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_cache(path: Path) -> Optional[PRData]:
    if not path.exists():
        return None
    try:
        return _from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.warning("Corrupt cache at %s: %s — re-fetching", path, exc)
        return None


# ── Fetcher ───────────────────────────────────────────────────────────────────

class GitHubFetcher:
    """Pulls PR history from a GitHub repo with concurrent fetching and local JSON cache."""

    def __init__(self, token: str, repo_str: str, cache_dir: Optional[Path] = None) -> None:
        self.gh = Github(token)
        self.repo = self.gh.get_repo(repo_str)
        self.repo_str = repo_str
        slug = repo_str.replace("/", "_")
        self.cache_dir = (cache_dir or _CACHE_DIR) / slug

    def fetch_prs(self, limit: int = 200) -> list[PRData]:
        """Fetch up to `limit` closed PRs, using cache where available, concurrent otherwise."""
        # Collect the lightweight PR stubs first (single paginated call)
        pr_stubs: list[PullRequest] = []
        for pr in self.repo.get_pulls(state="closed", sort="created", direction="desc"):
            if len(pr_stubs) >= limit:
                break
            pr_stubs.append(pr)

        cache_hits = 0
        results: dict[int, PRData] = {}

        # Split into cached vs needs-fetch
        to_fetch: list[PullRequest] = []
        for pr in pr_stubs:
            cached = _load_cache(_cache_path(self.cache_dir, pr.number))
            if cached is not None:
                results[pr.number] = cached
                cache_hits += 1
            else:
                to_fetch.append(pr)

        logger.info(
            "%d PRs from cache, %d to fetch from GitHub", cache_hits, len(to_fetch)
        )

        # Fetch the rest concurrently
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            future_to_pr = {pool.submit(self._extract_pr, pr): pr for pr in to_fetch}
            for future in as_completed(future_to_pr):
                pr = future_to_pr[future]
                try:
                    pr_data = future.result()
                    results[pr.number] = pr_data
                    _save_cache(_cache_path(self.cache_dir, pr.number), pr_data)
                    logger.debug("Fetched PR #%d: %s", pr.number, pr.title)
                except Exception as exc:
                    logger.warning("Skipping PR #%d: %s", pr.number, exc)

        # Return in original order (most-recent first)
        ordered = [results[pr.number] for pr in pr_stubs if pr.number in results]
        logger.info("Returning %d PRs from %s", len(ordered), self.repo_str)
        return ordered

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(RateLimitExceededException),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _extract_pr(self, pr: PullRequest) -> PRData:
        """Fetch all detail for one PR; the three sub-calls run concurrently."""
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_files = pool.submit(lambda: list(pr.get_files()))
            f_review = pool.submit(lambda: list(pr.get_review_comments()))
            f_comments = pool.submit(lambda: list(pr.get_issue_comments()))
            raw_files = f_files.result()
            raw_review = f_review.result()
            raw_comments = f_comments.result()

        files = [FileData(filename=f.filename, patch=f.patch) for f in raw_files]
        review_comments = [
            ReviewComment(
                body=rc.body,
                path=rc.path,
                line=rc.line,
                author=rc.user.login if rc.user else "unknown",
            )
            for rc in raw_review
        ]
        general_comments = [c.body for c in raw_comments if c.body]

        return PRData(
            pr_number=pr.number,
            title=pr.title or "",
            description=pr.body or "",
            labels=[label.name for label in pr.labels],
            state=pr.state,
            merged=pr.merged,
            author=pr.user.login if pr.user else "unknown",
            created_at=pr.created_at.isoformat() if pr.created_at else "",
            merged_at=pr.merged_at.isoformat() if pr.merged_at else None,
            files=files,
            review_comments=review_comments,
            general_comments=general_comments,
        )
