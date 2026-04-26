# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An AI-powered PR review agent. It ingests a GitHub repo's PR history into Qdrant, then runs a 4-agent pipeline to produce a cited code review for any PR diff. Exposed as a CLI and MCP server.

## Setup

```bash
conda activate pr-review-agent
docker compose up -d   # start Qdrant on port 6333
```

Requires `.env` with `GITHUB_TOKEN` and `ANTHROPIC_API_KEY`.

## Commands

```bash
python main.py ingest --repo owner/repo --limit 200
python main.py review --repo owner/repo --pr 42
python main.py serve   # start MCP server
python eval/evaluate.py --repo owner/repo
```

## Architecture

```
ingestion/          fetch → chunk → embed → Qdrant
retrieval/          vector_store.py, reranker.py
agents/             planner → executor → synthesizer → critic
mcp_server/         MCP tool: review_pr(diff, repo)
eval/               holdout eval harness
main.py             CLI entrypoint
```

## Key details

- Embedding model: `BAAI/bge-base-en-v1.5` (768-dim, local)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- LLM: `claude-haiku-4-5-20251001`
- Chunk types: `metadata`, `diff`, `review_comment`
- Chunk IDs are SHA-256 of `repo:pr:type:file` — upserts are idempotent
- GitHub fetcher caches PRs as JSON under `cache/` — safe to re-run
- PR fetching is concurrent (10 workers); upserts are batched + parallelized (4 workers)
