import datetime
from typing import Callable

from src.core.models import ToolCall

APPROVED_TOOLS = {"get_date", "lookup_employee"}

_DIRECTORY = {
    "alice": "alice@company.com | Engineering",
    "bob": "bob@company.com | HR",
    "carol": "carol@company.com | Finance",
}


def _get_date(_inputs: dict) -> str:
    return datetime.date.today().isoformat()


def _lookup_employee(inputs: dict) -> str:
    name = inputs.get("name", "").lower()
    return _DIRECTORY.get(name, f"Employee '{inputs.get('name', '')}' not found.")


_REGISTRY: dict[str, Callable] = {
    "get_date": _get_date,
    "lookup_employee": _lookup_employee,
}


class ToolGateway:
    """Executes approved tool calls; rejects anything not on the allowlist."""

    def execute(self, call: ToolCall) -> ToolCall:
        if call.name not in APPROVED_TOOLS:
            call.approved = False
            call.output = f"Tool '{call.name}' is not approved for use."
            return call

        call.approved = True
        call.output = _REGISTRY[call.name](call.inputs)
        return call

    def schema(self) -> list[dict]:
        return [
            {
                "name": "get_date",
                "description": "Returns today's date in ISO-8601 format.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "lookup_employee",
                "description": "Look up an employee's email and department by first name.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Employee's first name (e.g. 'Alice')",
                        }
                    },
                    "required": ["name"],
                },
            },
        ]
