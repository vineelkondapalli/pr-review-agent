"""Fetches PR history from a GitHub repository.

Uses the GitHub GraphQL API for PR metadata and comments (batched per page —
one GraphQL call per 100 PRs), and the REST API only for per-PR file patches
(one call per uncached PR).  This cuts total API calls ~3× versus the pure
REST approach and keeps ingestion well inside GitHub's 5 000 req/hour limit.
"""

import dataclasses
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_MAX_WORKERS = 5
_CACHE_DIR = Path("cache")
_GRAPHQL_URL = "https://api.github.com/graphql"

# PR authors matching these patterns are skipped before any REST calls are made.
_BOT_SUFFIXES = ("[bot]",)
_BOT_NAMES = frozenset({"github-actions", "dependabot", "renovate", "snyk-bot"})

_GRAPHQL_QUERY = """
query($owner: String!, $name: String!, $after: String, $first: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      states: [CLOSED, MERGED]
      first: $first
      after: $after
      orderBy: {field: CREATED_AT, direction: DESC}
    ) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title body state merged
        author { login }
        createdAt mergedAt
        labels(first: 20) { nodes { name } }
        reviewThreads(first: 50) {
          nodes {
            comments(first: 10) {
              nodes { body path originalLine author { login } }
            }
          }
        }
        comments(first: 50) { nodes { body } }
      }
    }
  }
}
"""


# ── Data classes ───────────────────────────────────────────────────────────────

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_bot_author(author: str) -> bool:
    lower = (author or "").lower()
    return any(lower.endswith(s) for s in _BOT_SUFFIXES) or lower in _BOT_NAMES


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


def _is_rate_limited(exc: BaseException) -> bool:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in (403, 429)
    return False


# ── Fetcher ────────────────────────────────────────────────────────────────────

class GitHubFetcher:
    """Fetches PR history using GraphQL (metadata + comments) + REST (file patches)."""

    def __init__(self, token: str, repo_str: str, cache_dir: Optional[Path] = None) -> None:
        self.token = token
        self.repo_str = repo_str
        self.owner, self.repo_name = repo_str.split("/", 1)
        slug = repo_str.replace("/", "_")
        self.cache_dir = (cache_dir or _CACHE_DIR) / slug
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def fetch_prs(self, limit: int | None = None) -> list[PRData]:
        """Fetch all PRs as a list (streams internally). Use stream_prs for large repos."""
        return [pr for batch in self.stream_prs(limit=limit) for pr in batch]

    def stream_prs(self, limit: int | None = None, batch_size: int = 100):
        """Yield batches of PRData.

        Each iteration makes 1 GraphQL call (metadata + comments for up to 100 PRs)
        then 1 REST call per uncached PR for file patches.
        Bots are filtered before any REST calls are made.
        """
        count = 0
        cursor: Optional[str] = None

        while True:
            page, has_next, cursor = self._graphql_page(cursor, min(batch_size, 100))

            if not page:
                break

            # Apply hard limit before bot filter so limit semantics are consistent
            if limit is not None:
                page = page[: max(0, limit - count)]
            count += len(page)

            # Skip bots — no REST call wasted on them
            to_resolve = [p for p in page if not _is_bot_author(p["author"])]

            if to_resolve:
                batch = self._resolve_files(to_resolve)
                if batch:
                    yield batch

            if not has_next or (limit is not None and count >= limit):
                break

    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=120),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_is_rate_limited),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _graphql_page(self, cursor: Optional[str], page_size: int):
        """Fetch one page of PRs (metadata + comments) via GraphQL.

        Returns (partial_prs, has_next, end_cursor).
        """
        resp = self._session.post(
            _GRAPHQL_URL,
            json={
                "query": _GRAPHQL_QUERY,
                "variables": {
                    "owner": self.owner,
                    "name": self.repo_name,
                    "after": cursor,
                    "first": page_size,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("data") is None:
            raise ValueError(f"GraphQL null data: {data.get('errors')}")
        if "errors" in data:
            raise ValueError(f"GraphQL errors: {data['errors']}")

        conn = data["data"]["repository"]["pullRequests"]
        page_info = conn["pageInfo"]

        partial_prs = []
        for node in conn["nodes"]:
            author = (node.get("author") or {}).get("login", "unknown")

            review_comments = []
            for thread in (node.get("reviewThreads") or {}).get("nodes", []):
                for comment in (thread.get("comments") or {}).get("nodes", []):
                    review_comments.append(ReviewComment(
                        body=comment.get("body", ""),
                        path=comment.get("path", ""),
                        line=comment.get("originalLine"),
                        author=(comment.get("author") or {}).get("login", "unknown"),
                    ))

            general_comments = [
                c["body"]
                for c in (node.get("comments") or {}).get("nodes", [])
                if c.get("body")
            ]

            partial_prs.append({
                "pr_number": node["number"],
                "title": node.get("title") or "",
                "description": node.get("body") or "",
                "labels": [lb["name"] for lb in (node.get("labels") or {}).get("nodes", [])],
                "state": (node.get("state") or "").lower(),
                "merged": node.get("merged", False),
                "author": author,
                "created_at": node.get("createdAt") or "",
                "merged_at": node.get("mergedAt"),
                "review_comments": review_comments,
                "general_comments": general_comments,
            })

        return partial_prs, page_info["hasNextPage"], page_info["endCursor"]

    def _resolve_files(self, partial_prs: list[dict]) -> list[PRData]:
        """Check cache, fetch file patches via REST for uncached PRs, assemble PRData."""
        results: dict[int, PRData] = {}
        to_fetch: list[dict] = []

        for partial in partial_prs:
            cached = _load_cache(_cache_path(self.cache_dir, partial["pr_number"]))
            if cached is not None:
                results[partial["pr_number"]] = cached
            else:
                to_fetch.append(partial)

        if to_fetch:
            with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
                futures = {pool.submit(self._fetch_files_rest, p): p for p in to_fetch}
                try:
                    for future in as_completed(futures, timeout=300):
                        partial = futures[future]
                        try:
                            files = future.result()
                            pr_data = PRData(
                                pr_number=partial["pr_number"],
                                title=partial["title"],
                                description=partial["description"],
                                labels=partial["labels"],
                                state=partial["state"],
                                merged=partial["merged"],
                                author=partial["author"],
                                created_at=partial["created_at"],
                                merged_at=partial["merged_at"],
                                files=files,
                                review_comments=partial["review_comments"],
                                general_comments=partial["general_comments"],
                            )
                            results[pr_data.pr_number] = pr_data
                            _save_cache(_cache_path(self.cache_dir, pr_data.pr_number), pr_data)
                            logger.debug("Fetched PR #%d", pr_data.pr_number)
                        except Exception as exc:
                            logger.warning("Skipping PR #%d: %s", partial["pr_number"], exc)
                except TimeoutError:
                    logger.warning("Batch timed out after 300 s; proceeding with partial results")

        return [results[p["pr_number"]] for p in partial_prs if p["pr_number"] in results]

    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=120),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_is_rate_limited),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _fetch_files_rest(self, partial_pr: dict) -> list[FileData]:
        """Fetch file patches for one PR via REST (the only per-PR REST call)."""
        files: list[FileData] = []
        url = f"https://api.github.com/repos/{self.repo_str}/pulls/{partial_pr['pr_number']}/files"
        while url:
            resp = self._session.get(url, params={"per_page": 100}, timeout=30)
            resp.raise_for_status()
            files.extend(
                FileData(filename=f["filename"], patch=f.get("patch"))
                for f in resp.json()
            )
            url = resp.links.get("next", {}).get("url")
        return files
