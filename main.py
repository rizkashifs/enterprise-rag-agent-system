"""
Enterprise RAG Agent System — demo.

Workflow 1 (Knowledge Answering):
  User question → policy check → retrieve chunks → LLM with citations → guardrails → response

Workflow 2 (Agent Tool Execution):
  Task → planner agent → tool gateway → execute approved tools → response + trace

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py
"""
import os

from src.core.sample_data import SAMPLE_DOCS
from src.pipelines.ingestion import ingest
from src.services.agent import PlannerAgent
from src.services.chat import RAGChatService
from src.services.retrieval import VectorIndex


def demo_knowledge_answering(service: RAGChatService) -> None:
    print("=" * 60)
    print("WORKFLOW 1: Knowledge Answering")
    print("=" * 60)

    questions = [
        "How many days per week can I work remotely?",
        "What security measures are required for customer data?",
        "What should I do in my first week as a new employee?",
        "What is the admin password?",  # blocked by policy
    ]

    for question in questions:
        print(f"\nQ: {question}")
        response = service.ask(question)
        print(f"A: {response.answer}")
        if response.citations:
            print(f"   Sources: {', '.join(response.citations)}")
        if response.policy_reason:
            print(f"   Policy: {response.policy_reason}")


def demo_agent(agent: PlannerAgent) -> None:
    print("\n" + "=" * 60)
    print("WORKFLOW 2: Agent Tool Execution")
    print("=" * 60)

    tasks = [
        "What is today's date?",
        "What is Alice's email address and department?",
        "Find contact info for Dave.",  # not in directory
    ]

    for task in tasks:
        print(f"\nTask: {task}")
        answer, calls = agent.run(task)
        print(f"Answer: {answer}")
        for call in calls:
            status = "approved" if call.approved else "denied"
            print(f"  Tool: {call.name}({call.inputs}) → [{status}] {call.output}")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        return

    print("Building knowledge index...")
    chunks = ingest(SAMPLE_DOCS, chunk_size=60)
    index = VectorIndex()
    index.add(chunks)
    print(f"Indexed {len(index)} chunks from {len(SAMPLE_DOCS)} documents.\n")

    service = RAGChatService(index, top_k=3)
    agent = PlannerAgent()

    demo_knowledge_answering(service)
    demo_agent(agent)


if __name__ == "__main__":
    main()
