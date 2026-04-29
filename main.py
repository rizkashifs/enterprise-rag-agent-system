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

from src.core.models import Document
from src.pipelines.ingestion import ingest
from src.services.agent import PlannerAgent
from src.services.chat import RAGChatService
from src.services.retrieval import VectorIndex

SAMPLE_DOCS = [
    Document(
        id="policy-001",
        content=(
            "Remote work policy: Employees may work remotely up to 3 days per week. "
            "Remote work requires written manager approval submitted at least 48 hours in advance. "
            "All remote workers must connect through the company VPN at all times. "
            "Employees in their first 90 days of employment must work on-site full-time. "
            "Equipment requests for remote work must be submitted through the IT helpdesk portal."
        ),
        source="HR Policy Manual v2.1",
    ),
    Document(
        id="security-001",
        content=(
            "Data classification policy: All customer data is classified as Confidential. "
            "Confidential data must be encrypted both at rest and in transit using AES-256. "
            "Access to customer data requires explicit written manager authorization. "
            "Customer data must never be stored on personal devices or shared via personal email. "
            "Violations of this policy must be reported to the security team within 24 hours."
        ),
        source="Security Policy v1.4",
    ),
    Document(
        id="onboarding-001",
        content=(
            "New employee onboarding checklist: Complete all HR paperwork on your first day. "
            "IT will provision your laptop and accounts within 24 hours of your start date. "
            "Complete mandatory security awareness training within your first 2 weeks. "
            "Schedule weekly check-ins with your manager during the first 90 days. "
            "Request access to internal systems through the IT helpdesk portal using your employee ID."
        ),
        source="Onboarding Guide 2024",
    ),
]


def demo_knowledge_answering(service: RAGChatService) -> None:
    print("=" * 60)
    print("WORKFLOW 1: Knowledge Answering")
    print("=" * 60)

    questions = [
        "How many days per week can I work remotely?",
        "What security measures are required for customer data?",
        "What should I do in my first week as a new employee?",
        "What is the admin password?",  # should be blocked by policy
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
