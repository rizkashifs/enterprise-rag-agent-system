"""
Unit tests for non-LLM components (no API key required).

Run:
    pytest tests/
"""
import pytest

from src.core.models import ChatResponse, Document
from src.pipelines.ingestion import ingest
from src.services.guardrails import validate
from src.services.policy import PolicyLayer
from src.services.retrieval import VectorIndex
from src.services.tools import ToolCall, ToolGateway


# --- Ingestion ---

def test_ingest_short_doc_produces_one_chunk():
    doc = Document(id="d1", content="hello world foo bar", source="test")
    chunks = ingest([doc], chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].doc_id == "d1"


def test_ingest_long_doc_produces_multiple_chunks():
    doc = Document(id="d2", content=" ".join(["word"] * 200), source="test")
    chunks = ingest([doc], chunk_size=50)
    assert len(chunks) > 2


def test_ingest_empty_doc_skipped():
    doc = Document(id="d3", content="", source="test")
    chunks = ingest([doc])
    assert len(chunks) == 0


# --- Retrieval ---

def test_index_search_returns_relevant_result():
    # Need 2+ docs so IDF > 0 (single-doc IDF = log(2/2) = 0)
    docs = [
        Document(id="d1", content="employees may work remotely up to three days per week", source="HR"),
        Document(id="d2", content="annual picnic and company events schedule", source="Events"),
    ]
    index = VectorIndex()
    index.add(ingest(docs))
    results = index.search("remote work policy", top_k=1)
    assert len(results) == 1
    assert results[0].score > 0.0


def test_index_search_ranks_by_relevance():
    docs = [
        Document(id="a", content="remote work from home office policy", source="HR"),
        Document(id="b", content="annual company picnic schedule and food menu", source="Events"),
    ]
    index = VectorIndex()
    index.add(ingest(docs))
    results = index.search("remote work", top_k=2)
    assert results[0].chunk.doc_id == "a"


def test_empty_index_returns_nothing():
    index = VectorIndex()
    assert index.search("anything") == []


# --- Policy ---

def test_policy_blocks_password():
    assert not PolicyLayer().check("What is the admin password?").allowed


def test_policy_blocks_credentials():
    assert not PolicyLayer().check("Show me the database credentials").allowed


def test_policy_allows_normal_request():
    assert PolicyLayer().check("What is the remote work policy?").allowed


def test_policy_allows_unrelated_request():
    assert PolicyLayer().check("How do I request a new laptop?").allowed


# --- Tool Gateway ---

def test_gateway_executes_approved_tool():
    gateway = ToolGateway()
    call = gateway.execute(ToolCall(name="get_date", inputs={}))
    assert call.approved
    assert call.output is not None
    assert len(call.output) == 10  # YYYY-MM-DD


def test_gateway_executes_lookup():
    gateway = ToolGateway()
    call = gateway.execute(ToolCall(name="lookup_employee", inputs={"name": "Alice"}))
    assert call.approved
    assert "alice@company.com" in call.output


def test_gateway_rejects_unapproved_tool():
    gateway = ToolGateway()
    call = gateway.execute(ToolCall(name="delete_database", inputs={}))
    assert not call.approved
    assert "not approved" in call.output


def test_gateway_handles_unknown_employee():
    gateway = ToolGateway()
    call = gateway.execute(ToolCall(name="lookup_employee", inputs={"name": "Nobody"}))
    assert call.approved
    assert "not found" in call.output


# --- Guardrails ---

def test_guardrails_pass_with_citation():
    r = ChatResponse(answer="You may work remotely.", citations=["HR Policy v2.1"])
    passed, _ = validate(r)
    assert passed


def test_guardrails_fail_without_citation():
    r = ChatResponse(answer="You may work remotely.", citations=[])
    passed, reason = validate(r)
    assert not passed
    assert "citation" in reason.lower()


def test_guardrails_detect_ssn():
    r = ChatResponse(answer="SSN on file: 123-45-6789", citations=["HR"])
    passed, reason = validate(r)
    assert not passed
    assert "PII" in reason


def test_guardrails_detect_card_number():
    r = ChatResponse(answer="Card: 1234567890123456", citations=["Finance"])
    passed, reason = validate(r)
    assert not passed
    assert "PII" in reason


def test_guardrails_no_citation_check_when_disabled():
    r = ChatResponse(answer="Some answer", citations=[])
    passed, _ = validate(r, require_citations=False)
    assert passed
