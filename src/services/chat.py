import anthropic

from src.core.models import ChatResponse, RetrievalResult
from src.services.guardrails import validate
from src.services.policy import PolicyLayer
from src.services.retrieval import VectorIndex

_SYSTEM_PROMPT = """You are an enterprise knowledge assistant.
Answer questions using only the provided context passages.
After each fact, cite the source in the format [Source: <name>].
If the context does not contain the answer, say "I don't have that information in the provided sources." """

_client = anthropic.Anthropic()


class RAGChatService:
    """Retrieval-augmented generation: policy check → retrieve → LLM → guardrails."""

    def __init__(self, index: VectorIndex, top_k: int = 3):
        self.index = index
        self.top_k = top_k
        self.policy = PolicyLayer()

    def ask(self, question: str) -> ChatResponse:
        # 1. Policy check
        decision = self.policy.check(question)
        if not decision.allowed:
            return ChatResponse(
                answer=f"Request blocked: {decision.reason}",
                citations=[],
                policy_reason=decision.reason,
            )

        # 2. Retrieve relevant chunks
        results: list[RetrievalResult] = self.index.search(question, self.top_k)

        # 3. Build context for the LLM
        context = "\n\n".join(
            f"[Passage {i + 1}] Source: {r.chunk.source}\n{r.chunk.content}"
            for i, r in enumerate(results)
        )

        # 4. Call Claude with a cached system prompt
        response = _client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }],
        )

        answer = next((b.text for b in response.content if b.type == "text"), "")
        relevant = [r for r in results if r.score > 0.01]
        citations = list({r.chunk.source for r in relevant})
        contexts = [r.chunk.content for r in relevant]

        chat_response = ChatResponse(answer=answer, citations=citations, contexts=contexts)

        # 5. Guardrails
        passed, reason = validate(chat_response)
        if not passed:
            chat_response.answer = f"[Response blocked by guardrails: {reason}]"

        return chat_response
