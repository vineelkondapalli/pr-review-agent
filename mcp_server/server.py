"""MCP server exposing the PR review pipeline as a single tool.

Add to Claude Code or Cursor with:
  {
    "mcpServers": {
      "pr-review": {
        "command": "python",
        "args": ["main.py", "serve"],
        "env": { "PYTHONPATH": "/path/to/pr-review-agent" }
      }
    }
  }
"""

import logging
import os

import anthropic
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from agents.critic import Critic
from agents.executor import Executor
from agents.planner import Planner
from agents.synthesizer import Synthesizer
from ingestion.embedder import Embedder
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore

load_dotenv()
logger = logging.getLogger(__name__)

server = Server("pr-review-agent")


def _build_pipeline(repo: str) -> tuple[Planner, Executor, Synthesizer, Critic]:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    collection = repo.replace("/", "_")
    vs = VectorStore(collection_name=collection)
    embedder = Embedder()
    reranker = Reranker()
    return (
        Planner(client),
        Executor(vs, embedder, reranker),
        Synthesizer(client),
        Critic(client),
    )


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="review_pr",
            description=(
                "Given a raw unified diff and a GitHub repo (owner/repo), runs the full "
                "PR review pipeline and returns a verified markdown code review with "
                "citations to past PRs in that repository. "
                "The repo must have been ingested first via `python main.py ingest --repo owner/repo`."
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
    logger.info("review_pr called for repo=%s, diff length=%d", repo, len(diff))

    try:
        planner, executor, synthesizer, critic = _build_pipeline(repo)

        plan = planner.plan(diff)
        logger.info("Plan: %d queries, files=%s", len(plan.queries), plan.file_filters)

        context_chunks = executor.execute(plan)
        logger.info("Retrieved %d context chunks", len(context_chunks))

        synthesis = synthesizer.synthesize(diff, context_chunks)
        final = critic.verify(synthesis.raw_markdown, context_chunks)

        if not final.verified:
            logger.warning("Critic removed hallucinated refs: %s", final.hallucinated_refs)

        return [TextContent(type="text", text=final.cleaned_review)]
    except Exception as exc:
        logger.error("review_pr failed: %s", exc, exc_info=True)
        return [TextContent(type="text", text=f"Error running review pipeline: {exc}")]


def run() -> None:
    import asyncio
    asyncio.run(stdio_server(server))
