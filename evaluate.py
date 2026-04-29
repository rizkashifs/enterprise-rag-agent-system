"""
RAGAS evaluation runner for the enterprise RAG pipeline.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python evaluate.py
    python evaluate.py --dataset examples/golden_dataset.json --output examples/eval_results.json
"""
import argparse
import json
import os

from src.core.sample_data import SAMPLE_DOCS
from src.pipelines.evaluation import run_evaluation
from src.pipelines.ingestion import ingest
from src.services.chat import RAGChatService
from src.services.retrieval import VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS + LLM judge.")
    parser.add_argument("--dataset", default="examples/golden_dataset.json")
    parser.add_argument("--output",  default="examples/eval_results.json")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        return

    # Build the same RAG system used in main.py
    print("Building RAG index...")
    chunks = ingest(SAMPLE_DOCS, chunk_size=60)
    index = VectorIndex()
    index.add(chunks)
    service = RAGChatService(index, top_k=3)
    print(f"Index ready: {len(index)} chunks from {len(SAMPLE_DOCS)} documents.\n")

    results = run_evaluation(service, args.dataset)

    # ── Print summary ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Samples evaluated : {results['num_samples']}")

    print("\nRAGAS metrics  (0 – 1, higher is better)")
    for metric, score in results["ragas_scores"].items():
        bar = "█" * round(score * 20)
        print(f"  {metric:<25}  {score:.3f}  {bar}")

    lc = results["llm_correctness"]
    print(f"\nLLM correctness judge  (0 – {lc['max']}, higher is better)")
    print(f"  Overall average      {lc['average']:.2f} / {lc['max']}")
    for cat, avg in lc["per_category"].items():
        print(f"  {cat:<25}  {avg:.2f}")

    print("\nPer-sample detail:")
    for s in results["samples"]:
        j = s.get("llm_judge", {})
        score = j.get("score", "?")
        reason = j.get("reason", "")
        print(f"  [{score}/5] {s['question'][:55]}")
        print(f"         ↳ {reason}")

    # ── Save full results ──────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
