"""PubMed/NCBI Entrez E-utilities client.

Two-step search:
  1. esearch  — get PubMed IDs (PMIDs) matching the query
  2. efetch   — fetch full records (title, authors, abstract, DOI, year) as XML

Optional: set NCBI_API_KEY in .env to raise rate limit from 3 to 10 req/s.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import httpx

from schemas.paper import Author, Paper

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_TIMEOUT = 30.0


def _api_key_param() -> dict:
    key = os.getenv("NCBI_API_KEY", "")
    return {"api_key": key} if key else {}


async def _esearch(client: httpx.AsyncClient, query: str, max_results: int) -> list[str]:
    """Return list of PMIDs for the query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": min(max_results, 200),
        "retmode": "json",
        **_api_key_param(),
    }
    resp = await client.get(_ESEARCH_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _parse_efetch_xml(xml_text: str) -> list[Paper]:
    """Parse PubMed efetch XML into Paper objects."""
    papers: list[Paper] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    for article in root.findall(".//PubmedArticle"):
        try:
            medline = article.find("MedlineCitation")
            if medline is None:
                continue
            art = medline.find("Article")
            if art is None:
                continue

            # Title
            title_el = art.find("ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else "Untitled"

            # Abstract
            abstract_parts = []
            for text_el in art.findall(".//AbstractText"):
                label = text_el.get("Label")
                text = "".join(text_el.itertext()).strip()
                if text:
                    abstract_parts.append(f"{label}: {text}" if label else text)
            abstract = " ".join(abstract_parts) or None

            # Authors
            authors: list[Author] = []
            for author_el in art.findall(".//Author"):
                last = author_el.findtext("LastName") or ""
                fore = author_el.findtext("ForeName") or author_el.findtext("Initials") or ""
                name = f"{fore} {last}".strip()
                if name:
                    authors.append(Author(name=name))

            # Year — try ArticleDate, then PubMedPubDate, then JournalIssue/PubDate
            year: int | None = None
            for date_el in art.findall("ArticleDate"):
                y = date_el.findtext("Year")
                if y and y.isdigit():
                    year = int(y)
                    break
            if not year:
                journal = art.find("Journal")
                if journal is not None:
                    pub_date = journal.find(".//PubDate")
                    if pub_date is not None:
                        y = pub_date.findtext("Year")
                        if y and y.isdigit():
                            year = int(y)

            # DOI
            doi: str | None = None
            for id_el in article.findall(".//ArticleId"):
                if id_el.get("IdType") == "doi":
                    doi = id_el.text.strip() if id_el.text else None
                    break

            # PMID for URL
            pmid_el = medline.find("PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    doi=doi,
                    arxiv_id=None,
                    url=url,
                    source="pubmed",
                    citation_count=None,  # PubMed does not expose citation counts
                )
            )
        except Exception:
            continue

    return papers


async def search(query: str, max_results: int = 20) -> list[Paper]:
    """
    Search PubMed and return up to `max_results` Paper objects.
    Returns an empty list on any HTTP or parsing error.
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            pmids = await _esearch(client, query, max_results)
            if not pmids:
                return []

            params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "rettype": "abstract",
                **_api_key_param(),
            }
            resp = await client.get(_EFETCH_URL, params=params)
            resp.raise_for_status()
            return _parse_efetch_xml(resp.text)
    except (httpx.HTTPError, Exception):
        return []
