"""IEEE Xplore REST API client.

Searches the IEEE Xplore digital library (6M+ technical documents).
Returns metadata + abstracts. Full-text requires IEEE subscription.
Free tier: 200 requests/day.

Requires: IEEE_API_KEY in .env
Register at: https://developer.ieee.org
"""
from __future__ import annotations

import os
import re

import httpx

from schemas.paper import Author, Paper

_BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
_TIMEOUT = 25.0


def _api_key() -> str:
    return os.getenv("IEEE_API_KEY", "")


async def search(query: str, max_results: int = 20) -> list[Paper]:
    """
    Search IEEE Xplore and return up to `max_results` Paper objects.
    Returns an empty list if IEEE_API_KEY is not set or on any error.
    """
    key = _api_key()
    if not key:
        return []

    params = {
        "querytext": query,
        "max_records": min(max_results, 200),
        "format": "json",
        "apikey": key,
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, Exception):
        return []

    articles = data.get("articles", [])
    papers: list[Paper] = []

    for item in articles:
        try:
            title = item.get("title") or "Untitled"

            # Authors — nested under "authors": {"authors": [...]}
            authors: list[Author] = []
            author_wrapper = item.get("authors") or {}
            for a in author_wrapper.get("authors", []):
                name = a.get("full_name") or a.get("author_url", "")
                if name and not name.startswith("http"):
                    authors.append(Author(name=name))

            # Year — publication_date can be "2023", "April 2023", or "2023-04"
            pub_date = item.get("publication_date") or ""
            year: int | None = None
            year_match = re.search(r"\b(\d{4})\b", pub_date)
            if year_match:
                year = int(year_match.group(1))

            abstract = item.get("abstract") or None
            doi = item.get("doi") or None

            # Build URL from article_number if no direct link
            article_number = item.get("article_number") or ""
            url = item.get("pdf_url") or (
                f"https://ieeexplore.ieee.org/document/{article_number}"
                if article_number else ""
            )

            # Citation count
            cited_raw = item.get("citing_paper_count")
            citation_count = int(cited_raw) if cited_raw is not None else None

            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    doi=doi,
                    arxiv_id=None,
                    url=url,
                    source="ieee",
                    citation_count=citation_count,
                )
            )
        except Exception:
            continue

    return papers
