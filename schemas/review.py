from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.paper import Paper


class ReviewSection(BaseModel):
    title: str
    content: str
    cited_papers: list[str] = Field(default_factory=list)  # DOIs or arxiv IDs


class LiteratureReview(BaseModel):
    topic: str
    generated_at: str  # ISO timestamp
    model_used: str
    total_papers_reviewed: int
    sections: list[ReviewSection]
    research_gaps: list[str] = Field(default_factory=list)
    future_directions: list[str] = Field(default_factory=list)
    bibliography: list[Paper] = Field(default_factory=list)


class NodeOutput(BaseModel):
    node: str
    data: dict


class AgentTrace(BaseModel):
    topic: str
    model_used: str
    start_time: str
    end_time: str
    search_queries: list[str] = Field(default_factory=list)
    papers_found: dict[str, int] = Field(default_factory=dict)  # source -> count
    papers_after_dedup: int = 0
    papers_selected: int = 0
    node_outputs: list[NodeOutput] = Field(default_factory=list)
    review: LiteratureReview | None = None
