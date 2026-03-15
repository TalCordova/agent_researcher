"""Elsevier Scopus Search API client.

Scopus indexes 90M+ records across Elsevier and non-Elsevier journals.
Returns metadata + abstracts; full-text retrieval requires ScienceDirect
institutional access (handled separately if needed).

Requires: ELSEVIER_API_KEY in .env
Register at: https://dev.elsevier.com
"""
from __future__ import annotations

import os
import re

import httpx

from schemas.paper import Author, Paper

_BASE_URL = "https://api.elsevier.com/content/search/scopus"
_TIMEOUT = 25.0
# Scopus field tags: https://dev.elsevier.com/sc_search_views.html
_FIELDS = ",".join([
    "dc:title",
    "dc:creator",
    "prism:coverDate",
    "dc:description",
    "prism:doi",
    "citedby-count",
    "prism:url",
    "author",
])


def _api_key() -> str:
    return os.getenv("ELSEVIER_API_KEY", "")


async def search(query: str, max_results: int = 20) -> list[Paper]:
    """
    Search Scopus and return up to `max_results` Paper objects.
    Returns an empty list if ELSEVIER_API_KEY is not set or on any error.
    """
    key = _api_key()
    if not key:
        return []

    # Scopus query syntax: TITLE-ABS-KEY searches title, abstract, and keywords
    params = {
        "query": f"TITLE-ABS-KEY({query})",
        "count": min(max_results, 200),
        "field": _FIELDS,
        "sort": "relevancy",
    }
    headers = {
        "X-ELS-APIKey": key,
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_BASE_URL, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, Exception):
        return []

    entries = data.get("search-results", {}).get("entry", [])
    papers: list[Paper] = []

    for item in entries:
        try:
            title = item.get("dc:title") or "Untitled"

            # Authors — Scopus returns a list of author objects or a single string
            authors: list[Author] = []
            raw_authors = item.get("author") or []
            if isinstance(raw_authors, list):
                for a in raw_authors:
                    name = a.get("authname") or (
                        f"{a.get('given-name', '')} {a.get('surname', '')}".strip()
                    )
                    if name:
                        authors.append(Author(name=name))
            elif isinstance(raw_authors, dict):
                name = raw_authors.get("authname", "")
                if name:
                    authors.append(Author(name=name))
            # Fallback: dc:creator is the first author string
            if not authors:
                creator = item.get("dc:creator", "")
                if creator:
                    authors.append(Author(name=creator))

            # Year from prism:coverDate (format: YYYY-MM-DD or YYYY)
            cover_date = item.get("prism:coverDate") or ""
            year: int | None = None
            year_match = re.match(r"(\d{4})", cover_date)
            if year_match:
                year = int(year_match.group(1))

            # Abstract — field is dc:description in Scopus
            abstract = item.get("dc:description") or None

            # DOI
            doi = item.get("prism:doi") or None

            # URL
            url = item.get("prism:url") or ""

            # Citation count
            cited_raw = item.get("citedby-count")
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
                    source="scopus",
                    citation_count=citation_count,
                )
            )
        except Exception:
            continue

    return papers
