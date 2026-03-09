"""LangGraph state definition for the Masters Student Agent."""
from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    # ---- Input ----
    topic: str
    max_papers: int
    model_name: str
    output_dir: str

    # ---- Accumulated during execution ----
    search_queries: list[str]
    raw_papers: list[dict]            # from all sources, pre-dedup
    deduplicated_papers: list[dict]   # after semantic dedup
    selected_papers: list[dict]       # after relevance filtering (top N)
    extraction_results: list[dict]    # key contributions + methodology per paper

    # LangGraph message accumulator (for LLM conversation history)
    messages: Annotated[list, add_messages]

    # Retry counter for the insufficient-results branch
    search_retry_count: int

    # ---- Output ----
    review: dict | None               # serialised LiteratureReview
    trace: dict[str, Any]             # accumulated trace for JSON output
    error: str | None
