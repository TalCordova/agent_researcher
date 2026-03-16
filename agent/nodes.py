"""
LangGraph node functions for the Masters Student Agent.

Each function receives the current ResearchState and returns a dict of
state fields to update. Nodes are pure functions — no side effects except
logging via display.py.

Workflow:
  plan_searches → execute_searches → deduplicate_filter
    → [conditional: widen_search or continue]
    → extract_key_info → synthesize_findings → write_review → save_outputs
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.state import ResearchState
from config import ModelConfig, get_model_config, json_mode_kwargs, litellm_kwargs
from display import (
    console,
    print_llm_call,
    print_papers_table,
    print_search_results,
)
from schemas.paper import Paper
from schemas.review import AgentTrace, LiteratureReview, NodeOutput, ReviewSection
from tools import arxiv_search, crossref, ieee, openalex, pubmed, scopus, semantic_scholar
from tools.embeddings import deduplicate_papers, score_relevance

litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_config(state: ResearchState) -> ModelConfig:
    return get_model_config(state["model_name"])


def _llm_call(
    config: ModelConfig,
    messages: list[dict],
    step: str,
    schema_hint: str = "",
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """
    Make a single LiteLLM completion call.
    Injects a JSON-schema system message for models without native JSON mode.
    Returns the assistant message text.
    """
    jm = json_mode_kwargs(config, schema_hint)
    use_json_prompt = "_json_prompt" in jm
    extra: dict[str, Any] = {}

    if use_json_prompt:
        # Prepend schema instruction as a system message
        schema_sys = {
            "role": "system",
            "content": (
                "You must respond with valid JSON only. "
                f"Required structure:\n{jm['_json_prompt']}"
            ),
        }
        messages = [schema_sys] + messages
    else:
        extra["response_format"] = jm.get("response_format", {"type": "json_object"})

    kwargs = litellm_kwargs(
        config,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra,
    )

    response = litellm.completion(**kwargs)
    text: str = response.choices[0].message.content or ""

    print_llm_call(step, messages[-1].get("content", "")[:200], text[:200])
    return text


def _parse_json(text: str) -> dict | list:
    """Extract JSON from LLM response, tolerating markdown code fences."""
    text = text.strip()
    # Strip ```json ... ``` wrappers
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def _papers_to_dicts(papers: list[Paper]) -> list[dict]:
    return [p.model_dump() for p in papers]


def _dicts_to_papers(dicts: list[dict]) -> list[Paper]:
    return [Paper(**d) for d in dicts]


# ---------------------------------------------------------------------------
# Node: plan_searches
# ---------------------------------------------------------------------------

def plan_searches(state: ResearchState) -> dict:
    topic = state["topic"]
    config = _get_config(state)

    schema_hint = '{"queries": ["query1", "query2", ...]}'
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research librarian. Generate specific, varied search queries "
                "for academic databases. Return JSON with a 'queries' array of 3-5 strings."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate 3-5 search queries to find academic papers about: {topic!r}\n"
                "Each query should target a different angle or subtopic. "
                "Optimise for academic paper databases (Semantic Scholar, arXiv, Crossref).\n"
                "Return JSON: {\"queries\": [\"...\", ...]}"
            ),
        },
    ]

    text = _llm_call(config, messages, step="plan_searches", schema_hint=schema_hint)
    try:
        parsed = _parse_json(text)
        queries: list[str] = parsed.get("queries", [topic]) if isinstance(parsed, dict) else [topic]
    except Exception:
        queries = [topic]

    console.print(f"  [dim]Search queries: {queries}[/dim]")

    return {
        "search_queries": queries,
        "trace": {
            **state.get("trace", {}),
            "search_queries": queries,
        },
    }


# ---------------------------------------------------------------------------
# Node: execute_searches
# ---------------------------------------------------------------------------

_SOURCES = ["semantic_scholar", "arxiv", "crossref", "openalex", "pubmed", "scopus", "ieee"]


async def _gather_searches(queries: list[str], max_per_query: int) -> dict[str, list[Paper]]:
    results: dict[str, list[Paper]] = {s: [] for s in _SOURCES}
    tasks = []
    for q in queries:
        tasks.append(semantic_scholar.search(q, max_per_query))
        tasks.append(arxiv_search.search(q, max_per_query))
        tasks.append(crossref.search(q, max_per_query))
        tasks.append(openalex.search(q, max_per_query))
        tasks.append(pubmed.search(q, max_per_query))
        tasks.append(scopus.search(q, max_per_query))
        tasks.append(ieee.search(q, max_per_query))

    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    per_query = len(_SOURCES)
    for i, query in enumerate(queries):
        base = i * per_query
        for j, source in enumerate(_SOURCES):
            r = all_results[base + j]
            if isinstance(r, list):
                results[source].extend(r)

    return results


def execute_searches(state: ResearchState) -> dict:
    queries = state.get("search_queries") or [state["topic"]]
    max_papers = state.get("max_papers", 20)
    max_per_query = max(5, max_papers // max(len(queries), 1))

    source_results = asyncio.run(_gather_searches(queries, max_per_query))

    all_papers: list[Paper] = []
    papers_found: dict[str, int] = {}
    for source, papers in source_results.items():
        papers_found[source] = len(papers)
        print_search_results(source, len(papers), [p.title for p in papers[:5]])
        all_papers.extend(papers)

    return {
        "raw_papers": _papers_to_dicts(all_papers),
        "trace": {
            **state.get("trace", {}),
            "papers_found": papers_found,
        },
    }


# ---------------------------------------------------------------------------
# Node: deduplicate_filter
# ---------------------------------------------------------------------------

def deduplicate_filter(state: ResearchState) -> dict:
    raw_dicts = state.get("raw_papers") or []
    if not raw_dicts:
        return {"deduplicated_papers": [], "selected_papers": [], "error": None}

    topic = state["topic"]
    max_papers = state.get("max_papers", 20)

    raw_papers = _dicts_to_papers(raw_dicts)
    console.print(f"  [dim]Raw papers: {len(raw_papers)}[/dim]")

    deduped = deduplicate_papers(raw_papers)
    console.print(f"  [dim]After semantic dedup: {len(deduped)}[/dim]")

    scored = score_relevance(deduped, topic)

    # Keep papers with relevance > 0.35, then take top max_papers
    filtered = [p for p in scored if (p.relevance_score or 0.0) >= 0.35]
    if not filtered:
        filtered = scored  # relax threshold if nothing passes

    selected = filtered[:max_papers]
    print_papers_table(selected, title="Selected papers")

    trace = state.get("trace", {})
    return {
        "deduplicated_papers": _papers_to_dicts(deduped),
        "selected_papers": _papers_to_dicts(selected),
        "trace": {
            **trace,
            "papers_after_dedup": len(deduped),
            "papers_selected": len(selected),
        },
        "error": None,
    }


# ---------------------------------------------------------------------------
# Node: widen_search
# ---------------------------------------------------------------------------

def widen_search(state: ResearchState) -> dict:
    """Broaden the search when too few papers are found. Max 2 retries."""
    topic = state["topic"]
    retry = state.get("search_retry_count", 0) + 1
    console.print(f"  [yellow]Widening search (attempt {retry})...[/yellow]")

    # Simpler, broader queries
    broad_queries = [
        topic,
        " ".join(topic.split()[:3]),  # first 3 words
        f"survey {topic}",
        f"review {topic}",
    ]

    return {
        "search_queries": broad_queries,
        "search_retry_count": retry,
        "raw_papers": [],
    }


# ---------------------------------------------------------------------------
# Node: extract_key_info
# ---------------------------------------------------------------------------

def extract_key_info(state: ResearchState) -> dict:
    selected_dicts = state.get("selected_papers") or []
    if not selected_dicts:
        return {"extraction_results": [], "selected_papers": []}

    config = _get_config(state)
    papers = _dicts_to_papers(selected_dicts)

    BATCH_SIZE = 5
    enriched: list[Paper] = []

    schema_hint = textwrap.dedent("""
    {"extractions": [
        {"index": 0, "key_contributions": ["...", "..."], "methodology": "..."},
        ...
    ]}
    """).strip()

    for batch_start in range(0, len(papers), BATCH_SIZE):
        batch = papers[batch_start : batch_start + BATCH_SIZE]

        papers_text = "\n\n".join(
            f"[{i}] Title: {p.title}\nAbstract: {(p.abstract or 'N/A')[:600]}"
            for i, p in enumerate(batch)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Extract structured information from paper abstracts. "
                    "Return JSON with an 'extractions' array."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"For each paper below, extract:\n"
                    f"- key_contributions: list of 2-4 bullet-point findings\n"
                    f"- methodology: one sentence describing the approach\n\n"
                    f"{papers_text}\n\n"
                    f"Return JSON: "
                    f'{{\"extractions\": [{{'
                    f'"index": 0, "key_contributions": ["..."], "methodology": "..."'
                    f"}}, ...]}}"
                ),
            },
        ]

        text = _llm_call(
            config, messages, step="extract_key_info",
            schema_hint=schema_hint, max_tokens=1024,
        )

        try:
            parsed = _parse_json(text)
            extractions = parsed.get("extractions", []) if isinstance(parsed, dict) else []
            for ex in extractions:
                idx = ex.get("index", 0)
                if 0 <= idx < len(batch):
                    batch[idx] = batch[idx].model_copy(update={
                        "key_contributions": ex.get("key_contributions", []),
                        "methodology": ex.get("methodology"),
                    })
        except Exception:
            pass  # keep original papers without extraction

        enriched.extend(batch)

    return {
        "selected_papers": _papers_to_dicts(enriched),
        "extraction_results": [
            {"title": p.title, "contributions": p.key_contributions, "methodology": p.methodology}
            for p in enriched
        ],
    }


# ---------------------------------------------------------------------------
# Node: synthesize_findings
# ---------------------------------------------------------------------------

def synthesize_findings(state: ResearchState) -> dict:
    papers = _dicts_to_papers(state.get("selected_papers") or [])
    topic = state["topic"]
    config = _get_config(state)

    papers_summary = "\n\n".join(
        f"Title: {p.title} ({p.year or 'n/d'})\n"
        f"Key contributions: {'; '.join(p.key_contributions) or 'N/A'}\n"
        f"Methodology: {p.methodology or 'N/A'}"
        for p in papers
    )

    schema_hint = textwrap.dedent("""
    {
      "themes": ["theme1", "theme2"],
      "methodological_trends": ["trend1"],
      "research_gaps": ["gap1", "gap2"],
      "future_directions": ["direction1"],
      "section_plan": [
        {"title": "Section Title", "paper_indices": [0, 1, 2], "summary": "..."}
      ]
    }
    """).strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert academic researcher synthesising a literature review. "
                "Analyse the provided papers and identify themes, trends, gaps, and structure. "
                "Return JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research topic: {topic!r}\n\n"
                f"Papers to synthesise:\n{papers_summary}\n\n"
                f"Provide a synthesis including:\n"
                f"- themes: major themes across the literature\n"
                f"- methodological_trends: common approaches\n"
                f"- research_gaps: what is missing or understudied\n"
                f"- future_directions: promising directions\n"
                f"- section_plan: 3-5 sections for the review, each with title, "
                f"paper_indices (which papers belong), and a brief summary\n\n"
                f"Return JSON matching: {schema_hint}"
            ),
        },
    ]

    text = _llm_call(config, messages, step="synthesize_findings", schema_hint=schema_hint, max_tokens=2048)
    try:
        synthesis = _parse_json(text)
    except Exception:
        synthesis = {"themes": [], "research_gaps": [], "future_directions": [], "section_plan": []}

    trace = state.get("trace", {})
    return {
        "trace": {
            **trace,
            "synthesis": synthesis,
        },
    }


# ---------------------------------------------------------------------------
# Node: write_review
# ---------------------------------------------------------------------------

def write_review(state: ResearchState) -> dict:
    papers = _dicts_to_papers(state.get("selected_papers") or [])
    topic = state["topic"]
    config = _get_config(state)
    synthesis = state.get("trace", {}).get("synthesis", {})

    # Build section plan context
    section_plan = synthesis.get("section_plan", [])
    if not section_plan:
        section_plan = [{"title": "Overview", "paper_indices": list(range(len(papers))), "summary": topic}]

    # Build per-section paper context
    sections_context = []
    for sec in section_plan:
        indices = sec.get("paper_indices", [])
        sec_papers = [papers[i] for i in indices if 0 <= i < len(papers)]
        refs = []
        for p in sec_papers:
            ref_id = p.doi or p.arxiv_id or p.title[:30]
            author = p.authors[0].name.split()[-1] if p.authors else "Unknown"
            refs.append(f"{author} ({p.year or 'n/d'}): {p.title}")
        sections_context.append({
            "title": sec.get("title", "Section"),
            "summary": sec.get("summary", ""),
            "refs": refs,
        })

    schema_hint = textwrap.dedent("""
    {
      "sections": [
        {
          "title": "Section Title",
          "content": "2-3 paragraphs of academic prose with inline citations like (Author, Year).",
          "cited_papers": ["doi:...", "arxiv:..."]
        }
      ],
      "research_gaps": ["gap1", "gap2"],
      "future_directions": ["direction1"]
    }
    """).strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert academic writer producing a literature review. "
                "Write clear, scholarly prose with inline citations. Return JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write a literature review on: {topic!r}\n\n"
                f"Research gaps identified: {synthesis.get('research_gaps', [])}\n"
                f"Future directions: {synthesis.get('future_directions', [])}\n\n"
                f"Sections to write:\n"
                + "\n".join(
                    f"- {s['title']}: {s['summary']}\n  Papers: {'; '.join(s['refs'][:5])}"
                    for s in sections_context
                )
                + f"\n\nFor each section, write 2-3 academic paragraphs with citations like (Author, Year). "
                f"Return JSON: {schema_hint}"
            ),
        },
    ]

    text = _llm_call(config, messages, step="write_review", schema_hint=schema_hint, max_tokens=8192)

    try:
        raw = _parse_json(text)
        if not isinstance(raw, dict):
            raise ValueError("Expected dict")
    except Exception as e:
        console.print(f"  [yellow]write_review parse error: {e} — falling back to synthesis data[/yellow]")
        raw = {
            "sections": [],
            "research_gaps": synthesis.get("research_gaps", []),
            "future_directions": synthesis.get("future_directions", []),
        }

    review = LiteratureReview(
        topic=topic,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model_used=state["model_name"],
        total_papers_reviewed=len(papers),
        sections=[
            ReviewSection(
                title=s.get("title", ""),
                content=s.get("content", ""),
                cited_papers=s.get("cited_papers", []),
            )
            for s in raw.get("sections", [])
        ],
        research_gaps=raw.get("research_gaps", synthesis.get("research_gaps", [])),
        future_directions=raw.get("future_directions", synthesis.get("future_directions", [])),
        bibliography=papers,
    )

    return {"review": review.model_dump()}


# ---------------------------------------------------------------------------
# Node: save_outputs
# ---------------------------------------------------------------------------

def save_outputs(state: ResearchState) -> dict:
    output_dir = Path(state.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_slug = re.sub(r"[^\w]+", "_", state["topic"].lower())[:40]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"{topic_slug}_{timestamp}"

    trace = state.get("trace", {})
    review_dict = state.get("review") or {}

    # Build full AgentTrace
    agent_trace = AgentTrace(
        topic=state["topic"],
        model_used=state["model_name"],
        start_time=trace.get("start_time", ""),
        end_time=datetime.now(timezone.utc).isoformat(),
        search_queries=trace.get("search_queries", []),
        papers_found=trace.get("papers_found", {}),
        papers_after_dedup=trace.get("papers_after_dedup", 0),
        papers_selected=trace.get("papers_selected", 0),
        node_outputs=[
            NodeOutput(node=k, data=v if isinstance(v, dict) else {"value": str(v)})
            for k, v in trace.items()
            if k not in ("start_time",)
        ],
        review=LiteratureReview(**review_dict) if review_dict else None,
    )

    # Write JSON trace
    json_path = str(base) + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agent_trace.model_dump(), f, indent=2, default=str)

    # Write Markdown review
    md_path = str(base) + ".md"
    if review_dict:
        _write_markdown(review_dict, md_path)

    return {
        "trace": {
            **trace,
            "json_path": json_path,
            "md_path": md_path,
        },
    }


def _write_markdown(review: dict, path: str) -> None:
    lines = [f"# Literature Review: {review.get('topic', '')}", ""]
    lines += [
        f"**Generated:** {review.get('generated_at', '')}  ",
        f"**Model:** {review.get('model_used', '')}  ",
        f"**Papers reviewed:** {review.get('total_papers_reviewed', 0)}",
        "",
    ]

    for sec in review.get("sections", []):
        lines.append(f"## {sec.get('title', 'Section')}")
        lines.append("")
        lines.append(sec.get("content", ""))
        lines.append("")
        if sec.get("cited_papers"):
            lines.append(f"*Citations: {', '.join(sec['cited_papers'])}*")
            lines.append("")

    gaps = review.get("research_gaps", [])
    if gaps:
        lines.append("## Research Gaps")
        lines.append("")
        for g in gaps:
            lines.append(f"- {g}")
        lines.append("")

    future = review.get("future_directions", [])
    if future:
        lines.append("## Future Directions")
        lines.append("")
        for d in future:
            lines.append(f"- {d}")
        lines.append("")

    bib = review.get("bibliography", [])
    if bib:
        lines.append("## Bibliography")
        lines.append("")
        for p in bib:
            authors = ", ".join(a.get("name", "") for a in p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
            year = p.get("year") or "n.d."
            doi = p.get("doi", "")
            doi_str = f" DOI: {doi}" if doi else ""
            lines.append(f"- {authors} ({year}). *{p.get('title', '')}*.{doi_str}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
