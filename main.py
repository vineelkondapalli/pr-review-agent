"""CLI entrypoint for the PR review agent.

Usage:
  python main.py ingest --repo owner/repo [--limit 200]
  python main.py review --repo owner/repo --pr 42 [--verbose]
  python main.py serve
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    from ingestion.github_fetcher import GitHubFetcher
    from ingestion.chunker import chunk_pr
    from ingestion.embedder import Embedder
    from retrieval.vector_store import VectorStore

    token = os.environ["GITHUB_TOKEN"]
    collection = args.repo.replace("/", "_")

    print(f"Fetching up to {args.limit} PRs from {args.repo}...")
    fetcher = GitHubFetcher(token=token, repo_str=args.repo)
    prs = fetcher.fetch_prs(limit=args.limit)
    print(f"Fetched {len(prs)} PRs")

    all_chunks = []
    for pr in prs:
        all_chunks.extend(chunk_pr(pr, repo=args.repo))
    print(f"Produced {len(all_chunks)} chunks")

    vs = VectorStore(collection_name=collection)
    embedder = Embedder()
    n = embedder.embed_chunks(all_chunks, vs)
    print(f"Upserted {n} new chunks (skipped {len(all_chunks) - n} existing)")


def cmd_review(args: argparse.Namespace) -> None:
    import anthropic
    from github import Auth, Github
    from ingestion.embedder import Embedder
    from retrieval.vector_store import VectorStore
    from retrieval.reranker import Reranker
    from agents.planner import Planner
    from agents.executor import Executor
    from agents.synthesizer import Synthesizer
    from agents.critic import Critic

    token = os.environ["GITHUB_TOKEN"]
    collection = args.repo.replace("/", "_")

    print(f"Fetching PR #{args.pr} from {args.repo}...")
    gh = Github(auth=Auth.Token(token))
    repo = gh.get_repo(args.repo)
    pr = repo.get_pull(args.pr)

    # Build diff from files
    diff_parts = []
    for f in pr.get_files():
        if f.patch:
            diff_parts.append(f"--- a/{f.filename}\n+++ b/{f.filename}\n{f.patch}")
    pr_diff = "\n\n".join(diff_parts)

    if not pr_diff.strip():
        print("No diff content found for this PR.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    vs = VectorStore(collection_name=collection)
    embedder = Embedder()
    reranker = Reranker()

    print("Planning retrieval queries...")
    planner = Planner(client)
    plan = planner.plan(pr_diff)
    print(f"Queries: {plan.queries}")

    print("Retrieving context...")
    executor = Executor(vs, embedder, reranker)
    context_chunks = executor.execute(plan)
    print(f"Retrieved {len(context_chunks)} context chunks")

    print("Synthesizing review...")
    synthesizer = Synthesizer(client)
    result = synthesizer.synthesize(pr_diff, context_chunks)

    print("Verifying citations...")
    critic = Critic(client)
    final = critic.verify(result.raw_markdown, context_chunks)

    if not final.verified:
        print(f"⚠ Critic removed hallucinated refs: {final.hallucinated_refs}")

    print("\n" + "=" * 60)
    print(f"PR #{args.pr}: {pr.title}")
    print("=" * 60)
    print(final.cleaned_review)


def cmd_serve(_args: argparse.Namespace) -> None:
    from mcp_server.server import run
    run()


def main() -> None:
    parser = argparse.ArgumentParser(description="PR Review Agent")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest PR history into Qdrant")
    p_ingest.add_argument("--repo", required=True, help="GitHub repo (owner/repo)")
    p_ingest.add_argument("--limit", type=int, default=200, help="Max PRs to ingest")

    p_review = sub.add_parser("review", help="Review a specific PR")
    p_review.add_argument("--repo", required=True, help="GitHub repo (owner/repo)")
    p_review.add_argument("--pr", required=True, type=int, help="PR number")

    sub.add_parser("serve", help="Start MCP server")

    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))

    dispatch = {"ingest": cmd_ingest, "review": cmd_review, "serve": cmd_serve}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
