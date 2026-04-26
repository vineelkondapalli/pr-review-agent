# PR Review Agent

An AI-powered code review agent that retrieves semantically similar past PRs, relevant code patterns, and historical review comments from a repository's history, then synthesizes a grounded, cited code review for any incoming PR diff.

Exposed as both a CLI and an MCP server — callable directly from Claude Code or Cursor mid-session.

---

## Architecture

```
GitHub Repo
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                  INGESTION PIPELINE                 │
│  GitHubFetcher → Chunker → Embedder → Qdrant       │
│  (PR history: metadata, diffs, review comments)    │
└─────────────────────────────────────────────────────┘
                         │
              (stored in Qdrant vector DB)
                         │
     ┌───────────────────▼─────────────────────┐
     │           AGENT PIPELINE                │
     │                                         │
     │  1. Planner  ──► sub-queries, filters   │
     │       │                                 │
     │  2. Executor ──► vector search +        │
     │       │          cross-encoder rerank   │
     │       │                                 │
     │  3. Synthesizer ──► grounded review     │
     │       │             with [PR #N] cites  │
     │       │                                 │
     │  4. Critic   ──► hallucination check    │
     │                  + citation cleanup     │
     └─────────────────────────────────────────┘
                         │
               Verified Markdown Review
                    ┌────┴────┐
                    ▼         ▼
                  CLI       MCP Server
                           (Claude Code / Cursor)
```

---

## Tech Stack

| Component       | Technology                                   |
|-----------------|----------------------------------------------|
| Language        | Python 3.11+                                 |
| GitHub API      | PyGithub                                     |
| Embeddings      | sentence-transformers `all-MiniLM-L6-v2`     |
| Vector DB       | Qdrant (local via Docker)                    |
| Reranking       | `cross-encoder/ms-marco-MiniLM-L-6-v2`       |
| Agent LLM       | Claude `claude-haiku-4-5-20251001`            |
| MCP Server      | official `mcp` Python SDK                    |
| Config          | python-dotenv                                |
| Eval            | custom pytest-based harness                  |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/vineelkondapalli/pr-review-agent.git
cd pr-review-agent

conda create -n pr-review-agent python=3.11
conda activate pr-review-agent
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker compose up -d
```

Qdrant dashboard: http://localhost:6333/dashboard

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your tokens:
#   GITHUB_TOKEN=ghp_...
#   ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

### Ingest a repository's PR history

```bash
python main.py ingest --repo owner/repo --limit 200
```

This fetches the last 200 closed PRs, chunks them into metadata/diff/review-comment chunks, embeds them with `all-MiniLM-L6-v2`, and upserts into Qdrant. Idempotent — safe to re-run.

### Review a specific PR

```bash
python main.py review --repo owner/repo --pr 42
```

Runs the full 4-agent pipeline (Planner → Executor → Synthesizer → Critic) and prints a markdown review with citations like `[PR #17]` and `[ref: auth.py]`.

### Start the MCP server

```bash
python main.py serve
```

---

## MCP Integration

### Claude Code

Add to your Claude Code MCP config (usually `~/.claude/claude_desktop_config.json` or the project `.mcp.json`):

```json
{
  "mcpServers": {
    "pr-review": {
      "command": "python",
      "args": ["main.py", "serve"],
      "cwd": "/absolute/path/to/pr-review-agent",
      "env": {
        "PYTHONPATH": "/absolute/path/to/pr-review-agent"
      }
    }
  }
}
```

Then in Claude Code, the `review_pr` tool is available:

> "Review this PR diff for owner/repo"

The tool signature:
```
review_pr(diff: str, repo: str) -> str
```

### Cursor

Add the same JSON block under `mcpServers` in your Cursor MCP settings.

---

## Evaluation

```bash
python eval/evaluate.py --repo owner/repo --holdout 30 --limit 200
```

Ingests all but the last 30 PRs, then runs the full pipeline on each holdout PR and reports:

| Metric             | Description                                                           |
|--------------------|-----------------------------------------------------------------------|
| Retrieval Recall   | Fraction of human-referenced PRs that appear in the top-20 chunks    |
| Citation Accuracy  | % of reviews where the Critic found no hallucinated citations         |
| Review Relevance   | LLM-as-judge score (1–5) for review specificity and usefulness        |

### Eval Results

| Repo | Recall | Citation Acc. | Avg Relevance |
|------|--------|---------------|---------------|
| *(run eval and fill in)* | | | |

---

## Project Structure

```
pr-review-agent/
├── ingestion/
│   ├── github_fetcher.py   # PyGithub wrapper with rate-limit retry
│   ├── chunker.py          # metadata / diff / review_comment chunks
│   └── embedder.py         # sentence-transformers + idempotent upsert
├── retrieval/
│   ├── vector_store.py     # Qdrant client wrapper
│   └── reranker.py         # cross-encoder reranking + dedup
├── agents/
│   ├── planner.py          # Claude → structured sub-queries (JSON)
│   ├── executor.py         # multi-query retrieval + rerank
│   ├── synthesizer.py      # Claude → grounded markdown review
│   └── critic.py           # Claude → citation verification + cleanup
├── mcp_server/
│   └── server.py           # MCP tool: review_pr(diff, repo)
├── eval/
│   └── evaluate.py         # holdout eval harness
├── main.py                 # CLI (ingest / review / serve)
├── docker-compose.yml      # Qdrant service
└── requirements.txt
```
