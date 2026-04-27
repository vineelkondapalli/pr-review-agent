"""MCP server exposing the PR review pipeline as a single tool.

Add to Claude Code or Cursor with:
  {
    "mcpServers": {
      "pr-review": {
        "command": "python",
        "args": ["main.py", "serve"],
        "cwd": "/absolute/path/to/pr-review-agent"
      }
    }
  }
"""

import asyncio
import logging
import os

# Configure file logging BEFORE importing models so model-loading log lines
# from models.py are captured rather than dropped by an unconfigured root logger.
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), "..", "mcp_server.log"),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

import anthropic
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

import models
from agents.critic import Critic
from agents.executor import Executor
from agents.planner import Planner
from agents.synthesizer import Synthesizer
from ingestion.embedder import Embedder
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore

load_dotenv()

server = Server("pr-review-agent")

# Cache VectorStore connections per collection so we don't reconnect on every call.
_vs_cache: dict[str, VectorStore] = {}

# Singletons initialized in run() after model pre-warm.
_client: anthropic.Anthropic | None = None
_embedder_inst: Embedder | None = None
_reranker_inst: Reranker | None = None

_MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}


def _resolve_model(model: str | None) -> str | None:
    if model is None:
        return None
    return _MODEL_ALIASES.get(model, model)


def _get_vs(collection: str) -> VectorStore:
    if collection not in _vs_cache:
        _vs_cache[collection] = VectorStore(collection_name=collection)
    return _vs_cache[collection]


def _build_pipeline(
    repo: str, model: str | None = None
) -> tuple[Planner, Executor, Synthesizer, Critic]:
    collection = repo.replace("/", "_")
    vs = _get_vs(collection)
    kwargs = {"model": model} if model else {}
    return (
        Planner(_client, **kwargs),
        Executor(vs, _embedder_inst, _reranker_inst),
        Synthesizer(_client, **kwargs),
        Critic(_client, **kwargs),
    )


def _ingest_repo(repo: str, limit: int | None) -> int:
    """Fetch, chunk, embed and upsert PRs for repo. Returns number of PRs ingested."""
    from ingestion.chunker import chunk_pr
    from ingestion.github_fetcher import GitHubFetcher

    token = os.environ["GITHUB_TOKEN"]
    collection = repo.replace("/", "_")
    vs = _get_vs(collection)
    fetcher = GitHubFetcher(token=token, repo_str=repo)
    total = 0
    for batch in fetcher.stream_prs(limit=limit):
        chunks: list = []
        for pr in batch:
            chunks.extend(chunk_pr(pr, repo=repo))
        _embedder_inst.embed_chunks(chunks, vs)
        total += len(batch)
    logger.info("Ingested %d PRs for %s", total, repo)
    return total


def _run_pipeline(diff: str, repo: str, model: str | None, ingest_limit: int | None) -> str:
    logger.info("_run_pipeline: start repo=%s", repo)
    collection = repo.replace("/", "_")
    vs = _get_vs(collection)

    try:
        info = vs.client.get_collection(collection)
        point_count = getattr(info, "points_count", None) or 0
    except Exception:
        point_count = 0
    logger.info("_run_pipeline: collection=%s points=%d", collection, point_count)

    if point_count == 0:
        logger.info("_run_pipeline: ingesting %s", repo)
        _ingest_repo(repo, ingest_limit)

    logger.info("_run_pipeline: building pipeline")
    planner, executor, synthesizer, critic = _build_pipeline(repo, model)

    logger.info("_run_pipeline: planning")
    plan = planner.plan(diff)
    logger.info("_run_pipeline: plan done, %d queries", len(plan.queries))

    logger.info("_run_pipeline: executing")
    context_chunks = executor.execute(plan)
    logger.info("_run_pipeline: retrieved %d chunks", len(context_chunks))

    logger.info("_run_pipeline: synthesizing")
    synthesis = synthesizer.synthesize(diff, context_chunks)

    logger.info("_run_pipeline: verifying")
    final = critic.verify(synthesis.raw_markdown, context_chunks)
    logger.info("_run_pipeline: done, verified=%s", final.verified)
    return final.cleaned_review


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="review_pr",
            description=(
                "Given a raw unified diff and a GitHub repo (owner/repo), runs the full "
                "PR review pipeline and returns a verified markdown code review with "
                "citations to past PRs in that repository. "
                "If the repo has not been ingested yet, it will be ingested automatically before review."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "diff": {
                        "type": "string",
                        "description": "The raw unified diff of the PR to review",
                    },
                    "repo": {
                        "type": "string",
                        "description": "GitHub repository in owner/repo format",
                    },
                    "model": {
                        "type": "string",
                        "description": (
                            "Claude model to use. Accepts aliases (haiku, sonnet, opus) "
                            "or full model IDs. Defaults to claude-sonnet-4-6."
                        ),
                    },
                    "ingest_limit": {
                        "type": "integer",
                        "description": (
                            "Max number of PRs to ingest if the repo has not been ingested yet. "
                            "Defaults to 200. Ignored if the repo is already ingested."
                        ),
                    },
                },
                "required": ["diff", "repo"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "review_pr":
        raise ValueError(f"Unknown tool: {name}")

    diff = arguments["diff"]
    repo = arguments["repo"]
    model = _resolve_model(arguments.get("model"))
    ingest_limit: int | None = arguments.get("ingest_limit", 200)
    logger.info("review_pr called for repo=%s, diff length=%d", repo, len(diff))

    await asyncio.to_thread(_ensure_singletons)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_run_pipeline, diff, repo, model, ingest_limit),
            timeout=600,
        )
        return [TextContent(type="text", text=result)]
    except asyncio.TimeoutError:
        logger.error("review_pr timed out after 300s for repo=%s", repo)
        return [TextContent(type="text", text="Error: review pipeline timed out after 5 minutes")]
    except Exception as exc:
        logger.error("review_pr failed: %s", exc, exc_info=True)
        return [TextContent(type="text", text=f"Error running review pipeline: {exc}")]


def _ensure_singletons() -> None:
    global _client, _embedder_inst, _reranker_inst
    if _client is not None:
        return
    logger.info("_ensure_singletons: creating Anthropic client")
    _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    logger.info("_ensure_singletons: creating Embedder/Reranker")
    _embedder_inst = Embedder()
    _reranker_inst = Reranker()
    logger.info("_ensure_singletons: done")


def run() -> None:
    # Load HF models synchronously on the main thread before serving any
    # requests. Background loading silently stalled inside the MCP server's
    # asyncio.to_thread worker, hanging the first tool call indefinitely.
    logger.info("run: loading models synchronously")
    models.load_sync()
    logger.info("run: models ready")

    async def _main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_main())
