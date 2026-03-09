# Masters Student Agent

An AI agent that performs end-to-end academic literature reviews. Give it a research topic and it returns a structured Markdown review backed by a full JSON execution trace.

## How it works

```
plan_searches → execute_searches → deduplicate_filter
    → [widen_search loop if < 3 papers, max 2 retries]
    → extract_key_info → synthesize_findings → write_review → save_outputs
```

1. **Plan** — LLM generates 3-5 targeted search queries from the topic
2. **Search** — Queries Semantic Scholar, arXiv, and Crossref in parallel
3. **Dedup & filter** — Local sentence-transformer embeddings (`all-MiniLM-L6-v2`) deduplicate and score relevance; papers below 0.35 cosine similarity to the topic are dropped
4. **Extract** — LLM extracts key contributions and methodology per paper (batched, 5 at a time)
5. **Synthesise** — LLM identifies themes, methodological trends, research gaps, and future directions across all papers
6. **Write** — LLM writes a structured academic review with inline citations
7. **Save** — Outputs a Markdown review and a full JSON execution trace to `outputs/`

## Quickstart

```bash
# Install dependencies (Python 3.12+ required)
pip install -r requirements.txt

# Copy and fill in API keys
cp .env.example .env   # or create .env manually (see below)

# Run with the default model (Claude)
python main.py "large language model agents"
```

## Usage

```bash
python main.py "research topic" [--model MODEL] [--max-papers N] [--output-dir DIR]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `claude` | Model to use (see table below) |
| `--max-papers` | `20` | Maximum papers to include in the review |
| `--output-dir` | `outputs/` | Directory for output files |

### Examples

```bash
# Cloud API models
python main.py "transformer architectures for protein folding"
python main.py "RAG for legal documents" --model gpt4mini
python main.py "protein folding" --model gpt4o

# Local via Ollama (pull model first)
ollama pull qwen2.5
python main.py "federated learning" --model qwen-ollama

# Local via vLLM (GPU, start server first — see below)
python main.py "diffusion models" --model qwen

# HuggingFace Inference API (cloud, no local GPU)
python main.py "attention mechanisms" --model hf-qwen --max-papers 30
```

## Supported models

| Key | Provider | Underlying model | Notes |
|-----|----------|-----------------|-------|
| `claude` | Anthropic API | claude-sonnet-4-6 | **Default.** Requires `ANTHROPIC_API_KEY` |
| `gpt4o` | OpenAI API | gpt-4o | Requires `OPENAI_API_KEY` |
| `gpt4mini` | OpenAI API | gpt-4o-mini | Cheaper. Requires `OPENAI_API_KEY` |
| `llama` | Ollama (local) | llama3.2 | `ollama pull llama3.2` |
| `mistral` | Ollama (local) | mistral | `ollama pull mistral` |
| `gemma` | Ollama (local) | gemma3 | `ollama pull gemma3` |
| `qwen-ollama` | Ollama (local) | qwen2.5 | `ollama pull qwen2.5` |
| `qwen` | vLLM (local GPU) | Qwen2.5-7B-Instruct | Start vLLM server first |
| `llama-vllm` | vLLM (local GPU) | Llama-3.1-8B-Instruct | Start vLLM server first |
| `hf-qwen` | HuggingFace API | Qwen2.5-72B-Instruct | Requires `HUGGINGFACE_API_KEY` |
| `hf-mistral` | HuggingFace API | Mistral-7B-Instruct-v0.3 | Requires `HUGGINGFACE_API_KEY` |

## Environment variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
HUGGINGFACE_API_KEY=hf_...

# Optional — these are the defaults
OLLAMA_BASE_URL=http://localhost:11434
VLLM_BASE_URL=http://localhost:8000/v1
CROSSREF_EMAIL=your@email.com   # enables Crossref polite pool (recommended)
```

## Running vLLM locally (GPU)

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# With 2 GPUs, serve a larger model
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000 --tensor-parallel-size 2

# Then run the agent
python main.py "your topic" --model qwen
```

## Output files

Each run writes two files to `outputs/`:

| File | Contents |
|------|----------|
| `{topic}_{timestamp}.md` | Human-readable literature review: sections with inline citations, research gaps, future directions, bibliography |
| `{topic}_{timestamp}.json` | Full `AgentTrace`: search queries, papers per source, dedup counts, relevance scores, per-paper extractions, synthesis JSON, final review |

## Project structure

```
agent_researcher/
├── main.py                    CLI entrypoint — argument parsing, graph invocation
├── config.py                  Model registry, LiteLLM routing, JSON-mode helpers
├── display.py                 Rich terminal helpers (tables, panels, streaming)
├── agent/
│   ├── graph.py               LangGraph StateGraph — nodes + conditional edges
│   ├── nodes.py               Node functions: plan, search, dedup, extract, synthesise, write, save
│   └── state.py               ResearchState TypedDict
├── tools/
│   ├── embeddings.py          sentence-transformers: semantic dedup + relevance scoring
│   ├── semantic_scholar.py    Semantic Scholar API client
│   ├── arxiv_search.py        arXiv API client
│   └── crossref.py            Crossref API client
├── schemas/
│   ├── paper.py               Paper, Author, PaperCollection Pydantic models
│   └── review.py              LiteratureReview, ReviewSection, AgentTrace models
└── outputs/                   Auto-created — JSON traces + Markdown reviews
```

## Extending the agent

### Adding a new model

1. Add an entry to `SUPPORTED_MODELS` in [config.py](config.py)
2. If it's a vLLM model, add its key to `_VLLM_MODELS`; for Ollama to `_OLLAMA_MODELS`; for HF to `_HF_MODELS`
3. Add any required API key to `.env`

LiteLLM handles the rest — no node code changes required.

### Adding a new search source

1. Create `tools/new_source.py` implementing `async def search(query: str, max_results: int) -> list[Paper]`
2. Import and call it in `agent/nodes.py` inside `_gather_searches()`

## Design decisions

- **LiteLLM** wraps all model calls — swap providers without touching node logic
- **LangGraph** manages state flow — `ResearchState` TypedDict is the single source of truth
- **sentence-transformers** handles dedup + relevance scoring locally (no LLM cost, GPU-accelerated)
- **JSON mode + Pydantic validation** on all LLM outputs — structured and type-safe
- **Search tools return `[]` on failure** — the agent degrades gracefully if a source is down
- **Rich streams node execution live** — full trace also saved to `outputs/` as JSON

## Requirements

Python 3.12+ and the packages in [requirements.txt](requirements.txt):

```
langgraph, langchain-core, litellm, pydantic, httpx,
arxiv, rich, python-dotenv, python-dateutil,
sentence-transformers, torch
```
