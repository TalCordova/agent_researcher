"""OpenAlex API client.

OpenAlex is a fully open catalogue of scholarly works (250M+ papers).
No API key required; polite pool is accessed by sending an email via the
User-Agent or mailto param.
"""
from __future__ import annotations

import os
import re

import httpx

from schemas.paper import Author, Paper

_BASE_URL = "https://api.openalex.org/works"
_TIMEOUT = 20.0


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """OpenAlex stores abstracts as an inverted index {word: [pos, ...]}.
    Reconstruct the original abstract string from it."""
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def _mailto_param() -> str:
    """Return polite-pool email if CROSSREF_EMAIL is set (reuses same env var)."""
    return os.getenv("CROSSREF_EMAIL", "")


async def search(query: str, max_results: int = 20) -> list[Paper]:
    """
    Search OpenAlex and return up to `max_results` Paper objects.
    Returns an empty list on any HTTP or parsing error.
    """
    params: dict = {
        "search": query,
        "per-page": min(max_results, 200),
        "select": "id,title,abstract_inverted_index,authorships,publication_year,doi,cited_by_count,primary_location",
    }
    email = _mailto_param()
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
    for item in data.get("results", []):
        try:
            # Authors
            authors: list[Author] = []
            for authorship in item.get("authorships", []):
                author_info = authorship.get("author") or {}
                name = author_info.get("display_name", "")
                if name:
                    authors.append(Author(name=name))

            # DOI — OpenAlex returns full URL like "https://doi.org/10.xxx/yyy"
            doi_raw = item.get("doi") or ""
            doi = re.sub(r"^https?://doi\.org/", "", doi_raw).strip() or None

            # Best URL
            primary = item.get("primary_location") or {}
            landing = primary.get("landing_page_url") or ""
            url = landing or item.get("id") or ""

            papers.append(
                Paper(
                    title=item.get("title") or "Untitled",
                    authors=authors,
                    year=item.get("publication_year"),
                    abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
                    doi=doi,
                    arxiv_id=None,
                    url=url,
                    source="openalex",
                    citation_count=item.get("cited_by_count"),
                )
            )
        except Exception:
            continue

    return papers
