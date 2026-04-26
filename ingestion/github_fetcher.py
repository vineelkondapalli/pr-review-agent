"""Fetches PR history from a GitHub repository using the PyGithub API."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from github import Github, GithubException, RateLimitExceededException
from github.PullRequest import PullRequest
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


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


class GitHubFetcher:
    """Pulls PR history from a GitHub repo, handling rate limits with exponential backoff."""

    def __init__(self, token: str, repo_str: str) -> None:
        self.gh = Github(token)
        self.repo = self.gh.get_repo(repo_str)
        self.repo_str = repo_str

    def fetch_prs(self, limit: int = 200) -> list[PRData]:
        """Fetch up to `limit` closed PRs (merged + unmerged) from the repo."""
        prs: list[PRData] = []
        count = 0

        for pr in self.repo.get_pulls(state="closed", sort="created", direction="desc"):
            if count >= limit:
                break
            try:
                pr_data = self._extract_pr(pr)
                prs.append(pr_data)
                count += 1
                logger.debug("Fetched PR #%d: %s", pr.number, pr.title)
            except Exception as exc:
                logger.warning("Skipping PR #%d due to error: %s", pr.number, exc)

        logger.info("Fetched %d PRs from %s", len(prs), self.repo_str)
        return prs

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(RateLimitExceededException),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _extract_pr(self, pr: PullRequest) -> PRData:
        files = [
            FileData(filename=f.filename, patch=f.patch)
            for f in pr.get_files()
        ]

        review_comments = [
            ReviewComment(
                body=rc.body,
                path=rc.path,
                line=rc.line,
                author=rc.user.login if rc.user else "unknown",
            )
            for rc in pr.get_review_comments()
        ]

        general_comments = [
            c.body for c in pr.get_issue_comments() if c.body
        ]

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
