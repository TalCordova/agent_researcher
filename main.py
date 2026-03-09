"""
Masters Student Agent — CLI entrypoint.

Usage:
    python main.py "research topic" [--model MODEL] [--max-papers N] [--output-dir DIR]

Examples:
    python main.py "transformer architectures for protein folding"
    python main.py "RAG for legal documents" --model gpt4mini
    python main.py "federated learning privacy" --model qwen-ollama
    python main.py "diffusion models" --model qwen   # requires vLLM server
    python main.py "attention mechanisms" --model hf-qwen --max-papers 30
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

from agent.graph import build_graph
from config import SUPPORTED_MODELS, get_model_config
from display import print_completion, print_error, print_header, stream_graph_events

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Masters Student Agent — AI-powered literature review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("topic", help="Research topic to review")
    parser.add_argument(
        "--model",
        default="claude",
        choices=list(SUPPORTED_MODELS.keys()),
        help=f"Model to use. Default: claude. Options: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=20,
        metavar="N",
        help="Maximum number of papers to include (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        metavar="DIR",
        help="Directory for output files (default: outputs/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate model config early (will raise with a helpful message if invalid)
    try:
        get_model_config(args.model)
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)

    print_header(args.topic, args.model)

    graph = build_graph()

    inputs: dict = {
        "topic": args.topic,
        "max_papers": args.max_papers,
        "model_name": args.model,
        "output_dir": args.output_dir,
        "search_queries": [],
        "raw_papers": [],
        "deduplicated_papers": [],
        "selected_papers": [],
        "extraction_results": [],
        "messages": [],
        "search_retry_count": 0,
        "review": None,
        "trace": {"start_time": datetime.now(timezone.utc).isoformat()},
        "error": None,
    }

    final_state = stream_graph_events(graph, inputs)

    if final_state.get("error"):
        print_error(final_state["error"])
        sys.exit(1)

    trace = final_state.get("trace", {})
    json_path = trace.get("json_path", "")
    md_path = trace.get("md_path", "")

    if json_path or md_path:
        print_completion(json_path, md_path)
    else:
        console.print("[yellow]Warning: no output files were written.[/yellow]")


if __name__ == "__main__":
    main()
