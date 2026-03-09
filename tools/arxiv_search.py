"""arXiv API client using the official `arxiv` Python package."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import arxiv

from schemas.paper import Author, Paper

_executor = ThreadPoolExecutor(max_workers=2)


def _sync_search(query: str, max_results: int) -> list[Paper]:
    """Blocking arXiv search — run in a thread to avoid blocking the event loop."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers: list[Paper] = []
    try:
        for result in client.results(search):
            doi = result.doi if result.doi else None
            arxiv_id = result.entry_id.split("/abs/")[-1] if result.entry_id else None
            papers.append(
                Paper(
                    title=result.title or "Untitled",
                    authors=[Author(name=a.name) for a in result.authors],
                    year=result.published.year if result.published else None,
                    abstract=result.summary,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    url=result.entry_id or "",
                    source="arxiv",
                    citation_count=None,  # arXiv does not provide citation counts
                )
            )
    except Exception:
        pass
    return papers


async def search(query: str, max_results: int = 20) -> list[Paper]:
    """Async wrapper around the blocking arXiv search."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _sync_search, query, max_results)
