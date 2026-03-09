from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Keys are the short names users pass via --model.
# Values are LiteLLM model strings (provider/model-id).
#
# vLLM models use the "openai/" prefix — LiteLLM routes them to the local
# vLLM server via VLLM_BASE_URL (default http://localhost:8000/v1).
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, str] = {
    # Cloud API
    "claude":       "anthropic/claude-sonnet-4-6",
    "gpt4o":        "openai/gpt-4o",
    "gpt4mini":     "openai/gpt-4o-mini",

    # Ollama — local, GPU-accelerated if CUDA is available
    "llama":        "ollama/llama3.2",
    "mistral":      "ollama/mistral",
    "gemma":        "ollama/gemma3",
    "qwen-ollama":  "ollama/qwen2.5",

    # vLLM — high-performance local GPU server (OpenAI-compatible)
    # Start with: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
    "qwen":         "openai/Qwen/Qwen2.5-7B-Instruct",
    "llama-vllm":   "openai/meta-llama/Llama-3.1-8B-Instruct",

    # HuggingFace Inference API — cloud, no local GPU needed
    "hf-qwen":      "huggingface/Qwen/Qwen2.5-72B-Instruct",
    "hf-mistral":   "huggingface/mistralai/Mistral-7B-Instruct-v0.3",
}

_VLLM_MODELS = {"qwen", "llama-vllm"}
_OLLAMA_MODELS = {"llama", "mistral", "gemma", "qwen-ollama"}
_HF_MODELS = {"hf-qwen", "hf-mistral"}

# Models that do NOT support native JSON mode — a schema prompt is injected instead.
_NO_JSON_MODE = _OLLAMA_MODELS | _HF_MODELS


@dataclass
class ModelConfig:
    model_name: str       # short key, e.g. "claude"
    litellm_model: str    # full LiteLLM string, e.g. "anthropic/claude-sonnet-4-6"
    api_base: str | None  # only set for vLLM/Ollama when a custom URL is needed
    supports_json_mode: bool


def get_model_config(model_name: str) -> ModelConfig:
    """Return a ModelConfig for the given short model name."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported: {', '.join(SUPPORTED_MODELS)}"
        )

    litellm_model = SUPPORTED_MODELS[model_name]
    api_base: str | None = None

    if model_name in _VLLM_MODELS:
        api_base = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    elif model_name in _OLLAMA_MODELS:
        api_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return ModelConfig(
        model_name=model_name,
        litellm_model=litellm_model,
        api_base=api_base,
        supports_json_mode=model_name not in _NO_JSON_MODE,
    )


def litellm_kwargs(config: ModelConfig, **extra) -> dict:
    """
    Build kwargs dict for litellm.completion().

    Merges api_base when needed and passes through any extra kwargs
    (e.g. messages, temperature, max_tokens).
    """
    kwargs: dict = {"model": config.litellm_model, **extra}
    if config.api_base:
        kwargs["api_base"] = config.api_base
    return kwargs


def json_mode_kwargs(config: ModelConfig, schema_description: str) -> dict:
    """
    Return response_format kwargs appropriate for the model.

    For models with native JSON mode: response_format={"type": "json_object"}.
    For others: inject a system message instructing JSON output.
    """
    if config.supports_json_mode:
        return {"response_format": {"type": "json_object"}}
    # Fallback: caller must prepend a system message with schema_description.
    # Return a sentinel so callers know to do this.
    return {"_json_prompt": schema_description}
