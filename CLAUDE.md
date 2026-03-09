# agent_researcher — Masters Student Agent

## What this project does

An AI agent that performs academic literature reviews. Given a research topic it:
1. Generates targeted search queries (via LLM)
2. Searches Semantic Scholar, arXiv, and Crossref in parallel
3. Deduplicates and scores paper relevance using local sentence-transformer embeddings (GPU-accelerated)
4. Extracts key contributions and methodology per paper (via LLM)
5. Synthesises themes, gaps, and trends across all papers (via LLM)
6. Writes a structured literature review (via LLM)
7. Saves a Markdown review and a full JSON execution trace to `outputs/`

## Running the agent

```bash
# Install dependencies first
pip install -r requirements.txt

# Default (Claude via Anthropic API)
python main.py "large language model agents"

# Other API models
python main.py "RAG for legal documents" --model gpt4mini
python main.py "protein folding" --model gpt4o

# Local via Ollama (pull model first: ollama pull qwen2.5)
python main.py "federated learning" --model qwen-ollama

# Local via vLLM (GPU, start server first — see below)
python main.py "diffusion models" --model qwen

# HuggingFace Inference API (cloud, no local GPU)
python main.py "attention mechanisms" --model hf-qwen --max-papers 30

# Options
python main.py --help
```

## Supported models

| Key | Provider | Notes |
|-----|----------|-------|
| `claude` | Anthropic API | Default. Requires `ANTHROPIC_API_KEY` |
| `gpt4o` | OpenAI API | Requires `OPENAI_API_KEY` |
| `gpt4mini` | OpenAI API | Cheaper. Requires `OPENAI_API_KEY` |
| `llama` | Ollama (local) | `ollama pull llama3.2` |
| `mistral` | Ollama (local) | `ollama pull mistral` |
| `gemma` | Ollama (local) | `ollama pull gemma3` |
| `qwen-ollama` | Ollama (local) | `ollama pull qwen2.5` |
| `qwen` | vLLM (local GPU) | Start vLLM server first (see below) |
| `llama-vllm` | vLLM (local GPU) | Start vLLM server first (see below) |
| `hf-qwen` | HuggingFace API | Requires `HUGGINGFACE_API_KEY` |
| `hf-mistral` | HuggingFace API | Requires `HUGGINGFACE_API_KEY` |

## Environment variables (.env)

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
HUGGINGFACE_API_KEY=hf_...
OLLAMA_BASE_URL=http://localhost:11434      # optional, this is the default
VLLM_BASE_URL=http://localhost:8000/v1     # optional, this is the default
CROSSREF_EMAIL=your@email.com              # optional, for Crossref polite pool
```

## Running vLLM locally (GPU)

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Then run the agent:
python main.py "your topic" --model qwen
```

With 2 GPUs you can serve larger models:
```bash
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000 --tensor-parallel-size 2
```

## Project structure

```
agent_researcher/
├── main.py              CLI entrypoint — argument parsing, graph invocation, display
├── config.py            Model registry, LiteLLM routing, JSON-mode helpers
├── display.py           Rich terminal helpers (tables, panels, streaming)
├── agent/
│   ├── graph.py         LangGraph StateGraph — nodes + conditional edges
│   ├── nodes.py         Node functions: plan, search, dedup, extract, synthesise, write, save
│   └── state.py         ResearchState TypedDict
├── tools/
│   ├── embeddings.py    sentence-transformers: semantic dedup + relevance scoring
│   ├── semantic_scholar.py   Semantic Scholar API client
│   ├── arxiv_search.py       arXiv API client
│   └── crossref.py           Crossref API client
├── schemas/
│   ├── paper.py         Paper, Author, PaperCollection Pydantic models
│   └── review.py        LiteratureReview, ReviewSection, AgentTrace models
└── outputs/             Auto-created — JSON traces + Markdown reviews
```

## Output files

Each run creates two files in `outputs/`:

- `{topic_slug}_{timestamp}.json` — Full `AgentTrace`: all intermediate steps (search queries, papers per source, dedup counts, relevance scores, per-paper extractions, synthesis JSON, final review)
- `{topic_slug}_{timestamp}.md` — Human-readable literature review with sections, inline citations, research gaps, future directions, and bibliography

## Key design decisions

- **LiteLLM** wraps all model calls — swap providers without changing node code
- **LangGraph** manages state flow — `ResearchState` TypedDict is the single source of truth
- **sentence-transformers** handles dedup + relevance scoring locally (no LLM cost, GPU-accelerated)
- **JSON mode + Pydantic validation** on all LLM outputs — structured and type-safe
- **Search tools return `[]` on failure** — the agent degrades gracefully if one source is down
- **Rich streams node execution live** — full trace also saved to `outputs/` as JSON

## Adding a new model

1. Add an entry to `SUPPORTED_MODELS` in [config.py](config.py)
2. If it's a vLLM model, add its key to `_VLLM_MODELS`; for Ollama to `_OLLAMA_MODELS`; for HF to `_HF_MODELS`
3. Add any required API key to `.env`
4. That is it — LiteLLM handles the rest

## Adding a new search source

1. Create `tools/new_source.py` implementing `async def search(query: str, max_results: int) -> list[Paper]`
2. Import and call it in `agent/nodes.py` inside `_gather_searches()`

## Future: RAG over papers

When ready to add retrieval-augmented synthesis:
- After `extract_key_info`, add a `build_vector_store` node using `sentence-transformers` + `chromadb` or `faiss`
- During `synthesize_findings`, retrieve relevant passages by query instead of passing all paper text
- Non-breaking change — adds a node and a tool file without modifying existing nodes

## Dependencies

See [requirements.txt](requirements.txt). Python 3.12+ required.
