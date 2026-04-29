from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    content: str
    source: str


@dataclass
class Chunk:
    id: str
    doc_id: str
    content: str
    source: str
    embedding: list[float] | None = None


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str


@dataclass
class ToolCall:
    name: str
    inputs: dict[str, Any]
    output: str | None = None
    approved: bool = False


@dataclass
class ChatResponse:
    answer: str
    citations: list[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    policy_reason: str = ""
