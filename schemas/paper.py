from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Author(BaseModel):
    name: str
    affiliations: list[str] = Field(default_factory=list)


class Paper(BaseModel):
    title: str
    authors: list[Author] = Field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str = ""
    source: Literal["semantic_scholar", "arxiv", "crossref", "openalex", "pubmed", "scopus", "ieee"]
    citation_count: int | None = None
    relevance_score: float | None = None  # 0.0–1.0, assigned by embeddings
    key_contributions: list[str] = Field(default_factory=list)  # extracted by LLM
    methodology: str | None = None  # extracted by LLM

    def dedup_key(self) -> str:
        """Canonical key for exact deduplication (DOI preferred, else normalised title)."""
        if self.doi:
            return f"doi:{self.doi.lower().strip()}"
        return f"title:{self.title.lower().strip()}"

    def text_for_embedding(self) -> str:
        parts = [self.title]
        if self.abstract:
            parts.append(self.abstract)
        return " ".join(parts)


class PaperCollection(BaseModel):
    papers: list[Paper]
    total_found: int
    query: str
    sources_searched: list[str]
