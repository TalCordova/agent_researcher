from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

if TYPE_CHECKING:
    from schemas.paper import Paper
    from schemas.review import ReviewSection

console = Console()


def print_header(topic: str, model: str) -> None:
    console.print(
        Panel(
            f"[bold white]{topic}[/bold white]\n[dim]Model: {model}[/dim]",
            title="[bold cyan]Masters Student Agent[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def print_node_start(node_name: str) -> None:
    label = node_name.replace("_", " ").title()
    console.rule(f"[bold cyan]{label}[/bold cyan]")


def print_search_results(source: str, count: int, sample_titles: list[str]) -> None:
    if count == 0:
        console.print(f"  [dim]{source}:[/dim] no results")
        return
    table = Table(
        title=f"[bold]{source}[/bold] — {count} paper(s) found",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Sample titles", style="dim", no_wrap=False)
    for title in sample_titles[:5]:
        table.add_row(title)
    console.print(table)


def print_papers_table(papers: list[Paper], title: str = "Papers") -> None:
    table = Table(
        title=title,
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Title", ratio=5)
    table.add_column("Year", justify="right", style="cyan", no_wrap=True)
    table.add_column("Source", style="magenta", no_wrap=True)
    table.add_column("Citations", justify="right", style="yellow", no_wrap=True)
    table.add_column("Relevance", justify="right", style="green", no_wrap=True)

    for p in papers:
        relevance = f"{p.relevance_score:.2f}" if p.relevance_score is not None else "-"
        citations = str(p.citation_count) if p.citation_count is not None else "-"
        year = str(p.year) if p.year else "-"
        table.add_row(p.title[:80], year, p.source, citations, relevance)

    console.print(table)


def print_llm_call(step: str, prompt_preview: str, response_preview: str) -> None:
    content = (
        f"[bold]Prompt:[/bold] {prompt_preview[:200].strip()!r}\n"
        f"[bold]Response:[/bold] {response_preview[:200].strip()!r}"
    )
    console.print(
        Panel(content, title=f"[dim]LLM call — {step}[/dim]", border_style="dim")
    )


def print_review_section(section: ReviewSection) -> None:
    md_text = f"## {section.title}\n\n{section.content}"
    console.print(Markdown(md_text))
    if section.cited_papers:
        console.print(f"  [dim]Citations: {', '.join(section.cited_papers)}[/dim]")


def print_error(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_completion(json_path: str, md_path: str) -> None:
    console.print(
        Panel(
            f"[green]JSON trace:[/green] {json_path}\n"
            f"[green]Markdown review:[/green] {md_path}",
            title="[bold green]Done[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


def stream_graph_events(graph, inputs: dict) -> dict:
    """
    Iterate over LangGraph streaming events, printing progress via Rich.
    Returns the final state dict.
    """
    from schemas.paper import Paper as PaperModel
    from schemas.review import ReviewSection as SectionModel

    final_state: dict = {}

    for event in graph.stream(inputs, stream_mode="updates"):
        for node_name, state_delta in event.items():
            print_node_start(node_name)

            # Show search results when available
            if "raw_papers" in state_delta and state_delta["raw_papers"]:
                raw = state_delta["raw_papers"]
                if isinstance(raw, list) and raw:
                    try:
                        papers = [PaperModel(**p) if isinstance(p, dict) else p for p in raw]
                        print_papers_table(papers, title=f"Raw papers after {node_name}")
                    except Exception:
                        console.print(f"  [dim]{len(raw)} raw papers found[/dim]")

            if "selected_papers" in state_delta and state_delta["selected_papers"]:
                raw = state_delta["selected_papers"]
                if isinstance(raw, list) and raw:
                    try:
                        papers = [PaperModel(**p) if isinstance(p, dict) else p for p in raw]
                        print_papers_table(papers, title="Selected papers (post-filter)")
                    except Exception:
                        console.print(f"  [dim]{len(raw)} papers selected[/dim]")

            if "review" in state_delta and state_delta["review"]:
                review = state_delta["review"]
                if isinstance(review, dict) and "sections" in review:
                    for sec in review["sections"][:2]:  # preview first 2 sections
                        try:
                            print_review_section(SectionModel(**sec))
                        except Exception:
                            pass

            if "error" in state_delta and state_delta["error"]:
                print_error(state_delta["error"])

            final_state.update(state_delta)

    return final_state
