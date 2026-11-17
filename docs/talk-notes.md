# DataHack talk — running notes

Scratch file. Add things as they happen; polish later.

Working title: *From one agent to a team: building (and measuring) a literature scout.*

## Story arc

1. The problem: keeping up with a field without drowning.
2. v1: the current LangGraph pipeline
   `plan_searches → execute_searches → deduplicate_filter → [widen_search?] → extract_key_info → synthesize_findings → write_review → save_outputs`
3. Why open APIs (arXiv, Semantic Scholar, OpenAlex, Crossref, PubMed) and not scraping Nature or Science.
4. How do we know it works? (evaluation — the core of the talk)
5. What breaks, and the split into roles (searcher / filter / summarizer / critic).
6. The same pipeline as Claude subagents: lines of code, cost, evaluation scores. Fixed workflow vs. model-orchestrated agent.

## Evaluation ideas

Hooks that already exist: every run writes a full `AgentTrace` JSON to `outputs/`, which can be the input to an evaluation harness.

- [ ] Gold set: a few topics, each with a hand-labeled list of must-find papers. Measure recall@N after dedup and after filtering.
- [ ] Precision of `selected_papers`: label each paper relevant or not.
- [ ] Recall per source: which source contributes unique hits? Can any source be dropped?
- [ ] Dedup quality: false merges vs. missed duplicates, across different similarity thresholds.
- [ ] `widen_search` loop: how often it fires, and does widening actually improve recall or only add noise?
- [ ] Faithfulness of extraction and review: are claims in the review supported by the abstract or paper? Check for hallucinated citations.
- [ ] Cost and latency per run, per model (LiteLLM makes swapping models easy, so compare them).
- [ ] Variance: same topic, several runs, how stable is the selected set?

## Failure cases seen

(Log real ones as they come up: topic, what went wrong, which node.)

- Silent source failures: search tools return `[]` on error, so a dead API looks like "no results". Worth surfacing in the trace and counting.
-

## Open questions

- Where does the split into multiple agents actually earn its keep, as opposed to just adding more nodes?
- Human in the loop: where should a researcher approve or redirect (e.g., after `plan_searches`)?
