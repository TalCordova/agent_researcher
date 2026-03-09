"""Crossref REST API client."""
from __future__ import annotations

import os

import httpx

from schemas.paper import Author, Paper

_BASE_URL = "https://api.crossref.org/works"
_TIMEOUT = 20.0


def _mailto() -> str:
    """Return polite-pool email param if CROSSREF_EMAIL is set."""
    email = os.getenv("CROSSREF_EMAIL", "")
    return email


async def search(query: str, max_results: int = 20) -> list[Paper]:
    """
    Search Crossref and return up to `max_results` Paper objects.
    Returns an empty list on any HTTP or parsing error.
    """
    params: dict = {
        "query": query,
        "rows": min(max_results, 100),
        "select": "DOI,title,abstract,author,published,is-referenced-by-count,URL",
    }
    email = _mailto()
    if email:
        params["mailto"] = email

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, Exception):
        return []

    papers: list[Paper] = []
    for item in data.get("message", {}).get("items", []):
        try:
            titles = item.get("title") or []
            title = titles[0] if titles else "Untitled"

            authors: list[Author] = []
            for a in item.get("author") or []:
                name_parts = [a.get("given", ""), a.get("family", "")]
                name = " ".join(p for p in name_parts if p).strip()
                authors.append(Author(name=name or "Unknown"))

            # Published year
            pub = item.get("published") or item.get("published-print") or {}
            date_parts = pub.get("date-parts", [[None]])
            year = date_parts[0][0] if date_parts and date_parts[0] else None

            # Abstract — Crossref often omits it
            abstract_raw = item.get("abstract", "") or ""
            # Strip JATS XML tags if present
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract_raw).strip() or None

            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    doi=item.get("DOI"),
                    arxiv_id=None,
                    url=item.get("URL") or "",
                    source="crossref",
                    citation_count=item.get("is-referenced-by-count"),
                )
            )
        except Exception:
            continue

    return papers
