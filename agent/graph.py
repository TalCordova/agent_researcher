"""LangGraph StateGraph definition for the Masters Student Agent."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    deduplicate_filter,
    execute_searches,
    extract_key_info,
    plan_searches,
    save_outputs,
    synthesize_findings,
    widen_search,
    write_review,
)
from agent.state import ResearchState

_MAX_RETRIES = 2


def _should_widen(state: ResearchState) -> str:
    """
    Conditional edge after deduplicate_filter.
    If fewer than 3 papers were selected and retries remain, widen the search.
    Otherwise proceed to extraction.
    """
    selected = state.get("selected_papers") or []
    retry_count = state.get("search_retry_count", 0)

    if len(selected) < 3 and retry_count < _MAX_RETRIES:
        return "widen_search"
    return "extract_key_info"


def build_graph() -> StateGraph:
    """Build and compile the research agent graph."""
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("plan_searches", plan_searches)
    graph.add_node("execute_searches", execute_searches)
    graph.add_node("deduplicate_filter", deduplicate_filter)
    graph.add_node("widen_search", widen_search)
    graph.add_node("extract_key_info", extract_key_info)
    graph.add_node("synthesize_findings", synthesize_findings)
    graph.add_node("write_review", write_review)
    graph.add_node("save_outputs", save_outputs)

    # Linear edges
    graph.add_edge(START, "plan_searches")
    graph.add_edge("plan_searches", "execute_searches")
    graph.add_edge("execute_searches", "deduplicate_filter")

    # Conditional: widen or continue
    graph.add_conditional_edges(
        "deduplicate_filter",
        _should_widen,
        {
            "widen_search": "widen_search",
            "extract_key_info": "extract_key_info",
        },
    )

    # Widen loops back to execute_searches (uses updated search_queries)
    graph.add_edge("widen_search", "execute_searches")

    # Remaining linear pipeline
    graph.add_edge("extract_key_info", "synthesize_findings")
    graph.add_edge("synthesize_findings", "write_review")
    graph.add_edge("write_review", "save_outputs")
    graph.add_edge("save_outputs", END)

    return graph.compile()
