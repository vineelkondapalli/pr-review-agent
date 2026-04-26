"""Revue — interactive CLI REPL for AI-powered PR reviews."""

from __future__ import annotations

import logging
import os
import pathlib
import sys
from typing import Any

# Ensure the project root is importable when running as an installed entry point.
# (sys.path already includes the cwd for `python main.py`, but not for `revue`.)
_PROJECT_ROOT = str(pathlib.Path(__file__).parent.resolve())
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.formatted_text import ANSI as PT_ANSI
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.status import Status
from rich.table import Table
from rich.text import Text

load_dotenv()


def _make_console() -> Console:
    """Create a Console that writes UTF-8 to the underlying buffer, bypassing Windows cp1252."""
    import io as _io
    if hasattr(sys.stdout, "buffer"):
        try:
            utf8_out = _io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
            return Console(file=utf8_out, legacy_windows=False)
        except Exception:
            pass
    return Console(legacy_windows=False)


console = _make_console()

# ── Color constants ────────────────────────────────────────────────────────────
PURPLE = "#8B5CF6"
CYAN = "#06B6D4"

BANNER = """\
 ██████╗ ███████╗██╗   ██╗██╗   ██╗███████╗
 ██╔══██╗██╔════╝██║   ██║██║   ██║██╔════╝
 ██████╔╝█████╗  ██║   ██║██║   ██║█████╗
 ██╔══██╗██╔══╝  ╚██╗ ██╔╝██║   ██║██╔══╝
 ██║  ██║███████╗ ╚████╔╝ ╚██████╔╝███████╗
 ╚═╝  ╚═╝╚══════╝  ╚═══╝   ╚═════╝ ╚══════╝"""

_VERDICT_STYLE = {
    "Approve": "bold green",
    "Nitpicks Only": "bold yellow",
    "Request Changes": "bold red",
}
_VERDICT_COLOR = {
    "Approve": "green",
    "Nitpicks Only": "yellow",
    "Request Changes": "red",
}


# ── Gradient helpers ───────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> str:
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _gradient_line(text: str) -> Text:
    """Apply a left-to-right purple→cyan gradient to a single line of text."""
    c1 = _hex_to_rgb(PURPLE)
    c2 = _hex_to_rgb(CYAN)
    result = Text()
    n = max(len(text) - 1, 1)
    for i, ch in enumerate(text):
        result.append(ch, style=_lerp_color(c1, c2, i / n))
    return result


# ── Banner ─────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    console.print()
    for line in BANNER.splitlines():
        console.print(_gradient_line(line), justify="left")
    console.print()
    subtitle = Text("AI-powered code review  •  RAG-grounded  •  MCP-enabled", style=CYAN, justify="center")
    console.print(Panel(subtitle, border_style=PURPLE, padding=(0, 4)))
    console.print(f"  [dim]version 1.0.0  •  powered by Claude[/dim]")
    console.print(f"  [dim]Type [bold]help[/bold] to get started or [bold]exit[/bold] to quit[/dim]")
    console.print()


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _make_prompt(session: dict[str, Any]) -> Text:
    active_repo = session.get("active_repo")
    prompt = Text()
    prompt.append("◆ ", style=f"{PURPLE} bold")
    prompt.append("revue", style=f"{PURPLE} bold")
    if active_repo:
        repo_name = active_repo.split("/")[-1]
        prompt.append(" [", style=PURPLE)
        prompt.append(repo_name, style=f"{CYAN} bold")
        prompt.append("]", style=PURPLE)
    prompt.append(" > ", style=f"{PURPLE} bold")
    return prompt


# ── Citations panel ────────────────────────────────────────────────────────────

def _citations_panel(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return

    # Extract titles from metadata chunks
    title_map: dict[int, str] = {}
    for chunk in chunks:
        if chunk.get("chunk_type") == "metadata":
            pr_num = chunk.get("pr_number")
            if pr_num is not None and pr_num not in title_map:
                first_line = chunk.get("text", "").split("\n")[0]
                prefix = f"PR #{pr_num}: "
                if first_line.startswith(prefix):
                    title_map[pr_num] = first_line[len(prefix):]

    # Aggregate best score per PR
    best_score: dict[int, float] = {}
    for chunk in chunks:
        pr_num = chunk.get("pr_number")
        if pr_num is None:
            continue
        score = float(chunk.get("rerank_score", chunk.get("score", 0.0)))
        if pr_num not in best_score or score > best_score[pr_num]:
            best_score[pr_num] = score

    table = Table(show_header=True, border_style="dim", padding=(0, 1))
    table.add_column("PR #", style=CYAN, width=6, no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Score", style="dim", justify="right", width=7, no_wrap=True)

    for pr_num in sorted(best_score, key=lambda n: best_score[n], reverse=True):
        title = title_map.get(pr_num, "[dim](no title)[/dim]")
        table.add_row(str(pr_num), title, f"{best_score[pr_num]:.3f}")

    console.print(Panel(table, title="[dim]Context Sources[/dim]", border_style="dim"))


# ── Commands ───────────────────────────────────────────────────────────────────

def _cmd_ingest(args: list[str], session: dict[str, Any]) -> None:
    if not args:
        console.print(f"[{PURPLE}]Usage:[/] ingest <owner/repo> [--limit N]")
        return

    repo = args[0]
    limit: int | None = None
    if "--limit" in args:
        idx = args.index("--limit")
        if idx + 1 < len(args):
            limit = int(args[idx + 1])

    from ingestion.github_fetcher import GitHubFetcher
    from ingestion.chunker import chunk_pr
    from ingestion.embedder import Embedder
    from retrieval.vector_store import VectorStore

    token = os.environ["GITHUB_TOKEN"]
    collection = repo.replace("/", "_")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Stage 1: Fetch
        fetch_label = f"[{CYAN}]Fetching PRs (limit: {limit})...[/]" if limit else f"[{CYAN}]Fetching all PRs...[/]"
        t_fetch = progress.add_task(fetch_label, total=None)
        fetcher = GitHubFetcher(token=token, repo_str=repo)
        prs = fetcher.fetch_prs(limit=limit)
        progress.update(t_fetch, description=f"[{CYAN}]Fetched {len(prs)} PRs[/]", total=1, completed=1)

        # Stage 2: Chunk
        t_chunk = progress.add_task(f"[{CYAN}]Chunking PRs...[/]", total=len(prs))
        all_chunks = []
        for pr in prs:
            all_chunks.extend(chunk_pr(pr, repo=repo))
            progress.advance(t_chunk)
        progress.update(t_chunk, description=f"[{CYAN}]Chunked → {len(all_chunks)} chunks[/]")

        # Stage 3: Embed (all-at-once encoding, indeterminate)
        vs = VectorStore(collection_name=collection)
        embedder = Embedder()
        approx_batches = max(1, (len(all_chunks) + 63) // 64)
        t_embed = progress.add_task(f"[{CYAN}]Embedding...[/]", total=None)
        t_upsert = progress.add_task(f"[{CYAN}]Upserting...[/]", total=approx_batches)

        def _on_upsert(done: int, total: int) -> None:
            progress.update(t_embed, total=1, completed=1,
                            description=f"[{CYAN}]Embedded ✓[/]")
            progress.update(t_upsert, completed=done, total=total,
                            description=f"[{CYAN}]Upserting...[/]")

        n = embedder.embed_chunks(all_chunks, vs, progress_callback=_on_upsert)
        # Mark embed done if no callback fired (e.g. all chunks existed)
        progress.update(t_embed, total=1, completed=1,
                        description=f"[{CYAN}]Embedded ✓[/]")
        progress.update(t_upsert, completed=approx_batches,
                        description=f"[{CYAN}]Upserted {n} new chunks[/]")

    console.print(
        f"[green]✓[/green] Ingested [bold]{len(prs)}[/bold] PRs "
        f"([bold]{len(all_chunks)}[/bold] chunks, [bold]{n}[/bold] new) "
        f"from [bold]{repo}[/bold]"
    )
    session["active_repo"] = repo
    session["active_collection"] = collection


def _cmd_review(args: list[str], session: dict[str, Any]) -> None:
    if not args:
        console.print(f"[{PURPLE}]Usage:[/] review <pr_number> | review <owner/repo> <pr_number>")
        return

    # Parse: review <N>  or  review owner/repo <N>
    if len(args) == 1:
        repo = session.get("active_repo")
        if not repo:
            console.print(f"[yellow]⚠ No active repo. Run 'ingest <owner/repo>' first.[/yellow]")
            return
        try:
            pr_number = int(args[0])
        except ValueError:
            console.print(f"[red]PR number must be an integer.[/red]")
            return
    else:
        repo = args[0]
        try:
            pr_number = int(args[1])
        except ValueError:
            console.print(f"[red]PR number must be an integer.[/red]")
            return

    collection = repo.replace("/", "_")

    # Guard: collection must exist before we can review
    from retrieval.vector_store import VectorStore
    known = {c["name"] for c in VectorStore.list_collections()}
    if collection not in known:
        console.print(
            f"[yellow]⚠ No ingested data for [bold]{repo}[/bold]. "
            f"Run 'ingest {repo}' first.[/yellow]"
        )
        return

    import anthropic
    from github import Auth, Github
    from ingestion.embedder import Embedder
    from retrieval.reranker import Reranker
    from agents.planner import Planner
    from agents.executor import Executor
    from agents.synthesizer import Synthesizer
    from agents.critic import Critic

    token = os.environ["GITHUB_TOKEN"]
    pr_title = ""

    with Status(f"[{CYAN}]Fetching diff...[/]", console=console) as status:
        gh = Github(auth=Auth.Token(token))
        gh_repo = gh.get_repo(repo)
        pr = gh_repo.get_pull(pr_number)
        pr_title = pr.title
        diff_parts = []
        for f in pr.get_files():
            if f.patch:
                diff_parts.append(f"--- a/{f.filename}\n+++ b/{f.filename}\n{f.patch}")
        pr_diff = "\n\n".join(diff_parts)

        if not pr_diff.strip():
            console.print("[red]No diff content found for this PR.[/red]")
            return

        status.update(f"[{CYAN}]Planning retrieval queries...[/]")
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        vs = VectorStore(collection_name=collection)
        embedder = Embedder()
        reranker = Reranker()
        planner = Planner(client)
        plan = planner.plan(pr_diff)

        status.update(f"[{CYAN}]Retrieving context ({len(plan.queries)} queries)...[/]")
        executor = Executor(vs, embedder, reranker)
        context_chunks = executor.execute(plan, pr_number_lt=pr_number)

        status.update(f"[{CYAN}]Synthesizing review...[/]")
        synthesizer = Synthesizer(client)
        result = synthesizer.synthesize(pr_diff, context_chunks)

        status.update(f"[{CYAN}]Verifying citations...[/]")
        critic = Critic(client)
        final = critic.verify(result.raw_markdown, context_chunks)

    # Header
    console.rule(f"[bold]PR #{pr_number}: {pr_title}[/bold]")
    console.print()

    # Verdict badge
    verdict = result.verdict
    vstyle = _VERDICT_STYLE.get(verdict, "bold white")
    vcolor = _VERDICT_COLOR.get(verdict, "white")
    console.print(Panel(
        Text(verdict, style=vstyle, justify="center"),
        title="Verdict",
        border_style=vcolor,
        width=30,
    ))
    console.print()

    if not final.verified:
        console.print(
            f"[yellow]⚠ Critic removed hallucinated refs: "
            f"{', '.join(final.hallucinated_refs)}[/yellow]"
        )
        console.print()

    # Review body
    console.print(Markdown(final.cleaned_review))
    console.print()

    # Citations
    _citations_panel(context_chunks)


def _cmd_chat(args: list[str], session: dict[str, Any]) -> None:
    repo = session.get("active_repo")
    if not repo:
        console.print(f"[yellow]⚠ No active repo. Run 'ingest <owner/repo>' first.[/yellow]")
        return

    collection = repo.replace("/", "_")

    import anthropic
    from ingestion.embedder import Embedder
    from retrieval.reranker import Reranker
    from retrieval.vector_store import VectorStore
    from agents.chat import ChatAgent

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    vs = VectorStore(collection_name=collection)
    embedder = Embedder()
    reranker = Reranker()
    agent = ChatAgent(client, vs, embedder, reranker)

    console.print(
        f"[dim]Chat with [{CYAN}]{repo}[/{CYAN}] PR history. "
        f"Type [bold]exit[/bold] or [bold]back[/bold] to return, [bold]reset[/bold] to clear history.[/dim]"
    )
    console.print()

    # Optional inline message: `chat what is the auth pattern?`
    pending: str | None = " ".join(args).strip() if args else None
    chat_pt = PromptSession(
        history=InMemoryHistory(),
        completer=_CHAT_COMPLETER,
    )
    _chat_prompt = PT_ANSI(f"\033[38;2;6;182;212m  chat > \033[0m")

    while True:
        if pending:
            question = pending
            pending = None
        else:
            try:
                question = chat_pt.prompt(_chat_prompt)
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            question = question.strip()
            if not question:
                continue

        low = question.lower()
        if low in ("exit", "back", "quit", "q"):
            break
        if low == "reset":
            agent.reset()
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        # Stream response
        buffer = ""
        source_chunks: list[dict[str, Any]] = []
        try:
            with Live(Markdown("▌"), console=console, refresh_per_second=12) as live:
                for delta, chunks in agent.stream_response(question):
                    if chunks is not None:
                        source_chunks = chunks
                    else:
                        buffer += delta
                        live.update(Markdown(buffer + "▌"))
        except Exception as exc:
            console.print(Panel(str(exc), title="Error", border_style="red"))
            continue

        # Render final markdown cleanly after Live exits
        console.print(Markdown(buffer))
        console.print()

        if source_chunks:
            # Show warning if all scores are very low
            best = max((c.get("rerank_score", c.get("score", 0.0)) for c in source_chunks), default=0.0)
            if best < 0.3:
                console.print(
                    "[yellow]⚠ Could not find relevant context for this question "
                    "in the ingested repo.[/yellow]"
                )
            else:
                _citations_panel(source_chunks)

        console.print()


def _cmd_collections(args: list[str], session: dict[str, Any]) -> None:
    from retrieval.vector_store import VectorStore

    try:
        cols = VectorStore.list_collections()
    except Exception as exc:
        console.print(Panel(f"Could not connect to Qdrant: {exc}", border_style="red"))
        return

    if not cols:
        console.print("[dim]No collections found in Qdrant.[/dim]")
        return

    active = session.get("active_collection")
    table = Table(title="Qdrant Collections", border_style=PURPLE, show_header=True)
    table.add_column("Collection", style=CYAN)
    table.add_column("Use as", style="dim")
    table.add_column("Vectors", justify="right", style="white")
    table.add_column("Status", style="green")

    for col in cols:
        name = col["name"]
        marker = " ◀" if name == active else ""
        # Convert collection name back to owner/repo hint (first underscore → slash)
        repo_hint = name.replace("_", "/", 1)
        table.add_row(name + marker, repo_hint, str(col["vectors_count"]), col["status"])

    console.print(table)


def _cmd_clear(args: list[str], session: dict[str, Any]) -> None:
    if not args:
        console.print(f"[{PURPLE}]Usage:[/] clear <owner/repo>")
        return

    repo = args[0]
    collection = repo.replace("/", "_")

    from retrieval.vector_store import VectorStore

    known = {c["name"] for c in VectorStore.list_collections()}
    if collection not in known:
        console.print(f"[red]Collection [bold]{collection}[/bold] does not exist.[/red]")
        return

    console.print(
        f"[yellow]Delete collection [bold]{collection}[/bold]? "
        f"This cannot be undone. (y/N) [/yellow]",
        end="",
    )
    sys.stdout.flush()
    try:
        confirm = sys.stdin.readline().strip().lower()
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        return

    if confirm != "y":
        console.print("[dim]Cancelled.[/dim]")
        return

    VectorStore(collection_name=collection).delete_collection()
    console.print(f"[green]✓ Cleared collection for [bold]{repo}[/bold].[/green]")

    if session.get("active_collection") == collection:
        session["active_repo"] = None
        session["active_collection"] = None


def _cmd_eval(args: list[str], session: dict[str, Any]) -> None:
    repo = session.get("active_repo")
    limit = 200
    holdout = 30

    i = 0
    while i < len(args):
        if args[i] == "--repo" and i + 1 < len(args):
            repo = args[i + 1]
            i += 2
        elif args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        elif args[i] == "--holdout" and i + 1 < len(args):
            holdout = int(args[i + 1])
            i += 2
        else:
            i += 1

    if not repo:
        console.print(
            f"[yellow]⚠ No active repo. Run 'ingest <owner/repo>' first "
            f"or specify: eval --repo owner/repo[/yellow]"
        )
        return

    from eval.evaluate import EvalRunner

    runner = EvalRunner(repo_str=repo, holdout=holdout, limit=limit)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        t = progress.add_task(f"[{CYAN}]Fetching & ingesting...[/]", total=holdout)

        def _on_eval(stage: str, done: int, total: int) -> None:
            if stage == "ingest":
                progress.update(t, description=f"[{CYAN}]Evaluating holdout PRs...[/]", total=total)
            elif stage == "eval":
                progress.update(t, completed=done, total=total,
                                description=f"[{CYAN}]Evaluating PR {done}/{total}...[/]")

        try:
            records = runner.run(progress_callback=_on_eval)
        except ValueError as exc:
            console.print(Panel(str(exc), border_style="red"))
            return

    if not records:
        console.print("[yellow]No eval records produced.[/yellow]")
        return

    table = Table(title=f"Eval Results: {repo}", border_style=PURPLE, show_header=True)
    table.add_column("PR #", justify="right", style=CYAN)
    table.add_column("Retrieval Recall", justify="right")
    table.add_column("Citation Accuracy", justify="center")
    table.add_column("Review Quality", justify="right")
    table.add_column("Hallucinations Caught", justify="center")

    for r in records:
        table.add_row(
            str(r.pr_number),
            f"{r.retrieval_recall:.2f}",
            "[green]✓[/green]" if r.citation_accurate else "[red]✗[/red]",
            str(r.relevance_score),
            "[green]✓[/green]" if r.citation_accurate else "[red]✗[/red]",
        )

    avg_recall = sum(r.retrieval_recall for r in records) / len(records)
    pct_cite = sum(1 for r in records if r.citation_accurate) / len(records) * 100
    avg_rel = sum(r.relevance_score for r in records) / len(records)

    table.add_section()
    table.add_row(
        f"[bold {CYAN}]AVG[/]",
        f"[bold {CYAN}]{avg_recall:.2f}[/]",
        f"[bold {CYAN}]{pct_cite:.0f}%[/]",
        f"[bold {CYAN}]{avg_rel:.2f}[/]",
        "",
    )

    console.print(table)


def _cmd_use(args: list[str], session: dict[str, Any]) -> None:
    if not args:
        console.print(f"[{PURPLE}]Usage:[/] use <owner/repo>")
        return

    repo = args[0]

    if "/" not in repo:
        console.print(
            f"[yellow]⚠ Expected [bold]owner/repo[/bold] format (e.g. [bold]encode/httpx[/bold]), "
            f"not a collection name.[/yellow]"
        )
        return

    collection = repo.replace("/", "_")

    from retrieval.vector_store import VectorStore

    known = {c["name"] for c in VectorStore.list_collections()}
    if collection not in known:
        console.print(
            f"[yellow]⚠ No ingested data for [bold]{repo}[/bold]. "
            f"Run 'ingest {repo}' first.[/yellow]"
        )
        return

    session["active_repo"] = repo
    session["active_collection"] = collection
    console.print(f"[green]✓ Active repo set to [bold]{repo}[/bold].[/green]")


def _cmd_serve(args: list[str], session: dict[str, Any]) -> None:
    from mcp_server.server import run
    console.print("[dim]Starting MCP server on stdio (Ctrl+C to stop)...[/dim]")
    run()


def _cmd_help(args: list[str], session: dict[str, Any]) -> None:
    table = Table(border_style=PURPLE, show_header=True, title="Revue Commands")
    table.add_column("Command", style=f"{CYAN} bold", no_wrap=True)
    table.add_column("Arguments", style="dim", no_wrap=True)
    table.add_column("Description", style="white")

    rows = [
        ("ingest", "<owner/repo> [--limit N]", "Ingest PR history (omit --limit for full repo)"),
        ("use", "<owner/repo>", "Set active repo (must already be ingested)"),
        ("review", "<pr_number>", "Review a PR (uses active repo)"),
        ("review", "<owner/repo> <pr_number>", "Review a PR in any repo"),
        ("chat", "[message]", "Chat about the repo's PR history (RAG-grounded)"),
        ("collections", "", "List all Qdrant collections"),
        ("clear", "<owner/repo>", "Delete a Qdrant collection"),
        ("eval", "[--repo R] [--limit N] [--holdout N]", "Run evaluation harness"),
        ("serve", "", "Start MCP server on stdio"),
        ("help", "", "Show this help table"),
        ("exit / quit", "", "Exit Revue"),
    ]
    for cmd, arg, desc in rows:
        table.add_row(cmd, arg, desc)

    console.print(table)


def _cmd_exit(args: list[str], session: dict[str, Any]) -> str:
    goodbye = _gradient_line("Goodbye. Happy shipping. 🚀")
    console.print(Panel(goodbye, border_style="dim", padding=(0, 4)))
    return "EXIT"


# ── REPL loop ──────────────────────────────────────────────────────────────────

COMMANDS: dict[str, Any] = {
    "ingest": _cmd_ingest,
    "use": _cmd_use,
    "review": _cmd_review,
    "chat": _cmd_chat,
    "collections": _cmd_collections,
    "clear": _cmd_clear,
    "eval": _cmd_eval,
    "serve": _cmd_serve,
    "help": _cmd_help,
    "exit": _cmd_exit,
    "quit": _cmd_exit,
}

# Commands whose second positional argument is an owner/repo
_REPO_COMMANDS = {"ingest", "use", "review", "clear"}


class _RevueCompleter(Completer):
    """Tab-completes command names and owner/repo arguments from ingested collections."""

    def __init__(self) -> None:
        self._repos: list[str] | None = None

    def invalidate(self) -> None:
        self._repos = None

    def _load_repos(self) -> list[str]:
        if self._repos is None:
            try:
                from retrieval.vector_store import VectorStore
                cols = VectorStore.list_collections()
                self._repos = [c["name"].replace("_", "/", 1) for c in cols]
            except Exception:
                self._repos = []
        return self._repos

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        words = text.split()
        at_space = text.endswith(" ")

        # First word: complete command names
        if not at_space and len(words) <= 1:
            partial = words[0] if words else ""
            for cmd in sorted(COMMANDS):
                if cmd.startswith(partial):
                    yield Completion(cmd, start_position=-len(partial))
            return

        if not words:
            return

        cmd = words[0].lower()
        if cmd not in _REPO_COMMANDS:
            return

        # Second positional word: complete owner/repo from ingested collections
        on_second = (at_space and len(words) == 1) or (not at_space and len(words) == 2)
        if not on_second:
            return

        partial = "" if at_space else words[1]
        for repo in self._load_repos():
            if repo.startswith(partial):
                yield Completion(repo, start_position=-len(partial))


_CHAT_COMPLETER = WordCompleter(["exit", "back", "reset", "quit"], ignore_case=True)


def _pt_main_prompt(session: dict[str, Any]) -> PT_ANSI:
    """Build the styled REPL prompt string for prompt_toolkit."""
    active_repo = session.get("active_repo")
    p = "\033[38;2;139;92;246m◆ revue"
    if active_repo:
        name = active_repo.split("/")[-1]
        p += f" [\033[38;2;6;182;212m{name}\033[38;2;139;92;246m]"
    p += " > \033[0m"
    return PT_ANSI(p)


def _repl(session: dict[str, Any]) -> None:
    completer = _RevueCompleter()
    pt = PromptSession(history=InMemoryHistory(), completer=completer)

    while True:
        try:
            line = pt.prompt(_pt_main_prompt(session))
        except KeyboardInterrupt:
            console.print("[dim]Use 'exit' to quit.[/dim]")
            continue
        except EOFError:
            _cmd_exit([], session)
            return

        line = line.strip()
        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()
        cmd_args = parts[1:]

        if cmd not in COMMANDS:
            console.print(
                f"[yellow]⚠ Unknown command '[bold]{cmd}[/bold]'. "
                f"Type [bold]help[/bold] for available commands.[/yellow]"
            )
            continue

        try:
            result = COMMANDS[cmd](cmd_args, session)
            if result == "EXIT":
                return
            if cmd in ("ingest", "clear"):
                completer.invalidate()
        except KeyboardInterrupt:
            console.print("[dim]Interrupted.[/dim]")
        except Exception as exc:
            console.print(
                Panel(
                    f"[red]{exc}[/red]",
                    title="[red bold]Error[/red bold]",
                    border_style="red",
                )
            )
            logging.getLogger(__name__).debug("Command error", exc_info=True)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Ensure stdout handles Unicode (needed on Windows where default is cp1252)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    _print_banner()
    session: dict[str, Any] = {"active_repo": None, "active_collection": None}
    _repl(session)


if __name__ == "__main__":
    main()
