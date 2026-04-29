"""
RAGAS evaluation pipeline with Claude as the LLM judge.

RAGAS metrics (all LLM-based, no embeddings required):
  - faithfulness:        Are the answer's claims supported by the retrieved context?
  - context_precision:   Are the retrieved contexts relevant to the question?
  - context_recall:      Does the retrieved context contain what the ground truth requires?

Custom metric (direct Claude call):
  - llm_correctness:     Score 0–5 comparing generated answer to ground truth.
"""
import json
import os

import anthropic
from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall, faithfulness

from src.services.chat import RAGChatService

_JUDGE_SYSTEM = (
    "You are a precise evaluator. "
    "Respond only with valid JSON containing 'score' (integer 0–5) and 'reason' (one sentence)."
)


def load_golden_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def collect_responses(service: RAGChatService, samples: list[dict]) -> list[dict]:
    """Run each golden-dataset question through the RAG system."""
    enriched = []
    for i, sample in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] {sample['question'][:65]}...")
        response = service.ask(sample["question"])
        enriched.append({
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "answer": response.answer,
            "contexts": response.contexts if response.contexts else ["No context retrieved."],
            "category": sample.get("category", ""),
        })
    return enriched


def _ragas_llm() -> LangchainLLMWrapper:
    """Claude via LangChain adapter for RAGAS metric evaluation."""
    llm = ChatAnthropic(
        model="claude-opus-4-7",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    return LangchainLLMWrapper(llm)


def run_ragas_metrics(enriched: list[dict]) -> dict[str, float]:
    """
    Evaluate retrieved contexts and generated answers using RAGAS.
    Returns a dict of metric_name → float score (0–1).
    """
    ragas_llm = _ragas_llm()
    metrics = [faithfulness, context_precision, context_recall]
    for m in metrics:
        m.llm = ragas_llm

    dataset = Dataset.from_dict({
        "question":     [s["question"] for s in enriched],
        "answer":       [s["answer"] for s in enriched],
        "contexts":     [s["contexts"] for s in enriched],
        "ground_truth": [s["ground_truth"] for s in enriched],
    })

    result = evaluate(dataset, metrics=metrics)
    return {k: round(float(v), 4) for k, v in dict(result).items()}


def llm_judge_correctness(sample: dict, client: anthropic.Anthropic) -> dict:
    """
    Use Claude directly to judge whether the generated answer is correct.

    Scale:
      5 = Fully correct and complete
      4 = Mostly correct, minor omissions
      3 = Partially correct
      2 = Mostly incorrect
      1 = Marginally relevant but wrong
      0 = Wrong, off-topic, or blocked by policy

    Returns {"score": int, "reason": str}.
    """
    prompt = (
        f"Question: {sample['question']}\n"
        f"Ground truth: {sample['ground_truth']}\n"
        f"Generated answer: {sample['answer']}\n\n"
        "Rate the generated answer for correctness on the 0–5 scale above. "
        'Return JSON: {"score": <integer>, "reason": "<one sentence>"}'
    )
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=150,
        system=[{"type": "text", "text": _JUDGE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "{}")
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"score": -1, "reason": "Could not parse judge response."}


def run_evaluation(
    service: RAGChatService,
    golden_dataset_path: str = "examples/golden_dataset.json",
) -> dict:
    """
    Full pipeline:
      1. Load golden dataset
      2. Collect RAG responses (answer + retrieved contexts)
      3. Score with RAGAS (faithfulness, context_precision, context_recall)
      4. Score with custom LLM correctness judge
      5. Return combined results dict
    """
    samples = load_golden_dataset(golden_dataset_path)
    print(f"Loaded {len(samples)} samples from {golden_dataset_path}\n")

    print("Step 1/3  Collecting RAG responses...")
    enriched = collect_responses(service, samples)

    print("\nStep 2/3  Running RAGAS metrics (LLM as judge via LangChain-Anthropic)...")
    ragas_scores = run_ragas_metrics(enriched)

    print("\nStep 3/3  Running custom LLM correctness judge (direct Anthropic SDK)...")
    client = anthropic.Anthropic()
    for i, sample in enumerate(enriched, 1):
        print(f"  [{i}/{len(enriched)}] judging...")
        sample["llm_judge"] = llm_judge_correctness(sample, client)

    scores = [s["llm_judge"]["score"] for s in enriched if s["llm_judge"]["score"] >= 0]
    avg_correctness = round(sum(scores) / len(scores), 2) if scores else 0.0

    return {
        "ragas_scores": ragas_scores,
        "llm_correctness": {
            "average": avg_correctness,
            "max": 5,
            "per_category": _scores_by_category(enriched),
        },
        "num_samples": len(enriched),
        "samples": enriched,
    }


def _scores_by_category(enriched: list[dict]) -> dict[str, float]:
    by_cat: dict[str, list[int]] = {}
    for s in enriched:
        cat = s.get("category", "unknown")
        score = s.get("llm_judge", {}).get("score", -1)
        if score >= 0:
            by_cat.setdefault(cat, []).append(score)
    return {cat: round(sum(v) / len(v), 2) for cat, v in by_cat.items()}
