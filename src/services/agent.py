import anthropic

from src.core.models import ToolCall
from src.services.tools import ToolGateway

_SYSTEM_PROMPT = """You are an enterprise task agent. Use the available tools to answer questions.
Be concise. Only call a tool when you need real data you don't have."""

_client = anthropic.Anthropic()


class PlannerAgent:
    """Agentic loop: plan → call tools via gateway → respond."""

    def __init__(self):
        self.gateway = ToolGateway()

    def run(self, task: str) -> tuple[str, list[ToolCall]]:
        """Run the agent. Returns (final_answer, list_of_tool_calls_made)."""
        messages = [{"role": "user", "content": task}]
        tools = self.gateway.schema()
        calls_made: list[ToolCall] = []

        for _ in range(5):  # safety cap
            response = _client.messages.create(
                model="claude-opus-4-7",
                max_tokens=512,
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=tools,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                answer = next((b.text for b in response.content if b.type == "text"), "")
                return answer, calls_made

            # Execute each tool_use block and collect results
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                call = ToolCall(name=block.name, inputs=block.input)
                call = self.gateway.execute(call)
                calls_made.append(call)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": call.output or "",
                })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        return "Agent reached the iteration limit.", calls_made
