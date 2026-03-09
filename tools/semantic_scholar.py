"""Semantic Scholar Academic Graph API client."""
from __future__ import annotations

import httpx

from schemas.paper import Author, Paper

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "paperId,title,abstract,authors,year,citationCount,externalIds,url"
_TIMEOUT = 20.0


async def search(query: str, max_results: int = 20) -> list[Paper]:
    """
    Search Semantic Scholar and return up to `max_results` Paper objects.
    Returns an empty list on any HTTP or parsing error.
    """
    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": _FIELDS,
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, Exception):
        return []

    papers: list[Paper] = []
    for item in data.get("data", []):
        try:
            external_ids = item.get("externalIds") or {}
            papers.append(
                Paper(
                    title=item.get("title") or "Untitled",
                    authors=[
                        Author(name=a.get("name", ""))
                        for a in item.get("authors", [])
                    ],
                    year=item.get("year"),
                    abstract=item.get("abstract"),
                    doi=external_ids.get("DOI"),
                    arxiv_id=external_ids.get("ArXiv"),
                    url=item.get("url") or "",
                    source="semantic_scholar",
                    citation_count=item.get("citationCount"),
                )
            )
        except Exception:
            continue

    return papers
