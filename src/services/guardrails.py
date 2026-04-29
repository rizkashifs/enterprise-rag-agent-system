import re

from src.core.models import ChatResponse

# Matches SSN (123-45-6789) or 16-digit card numbers
_PII_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b|\b\d{16}\b')


def validate(response: ChatResponse, require_citations: bool = True) -> tuple[bool, str]:
    """Return (passed, reason). Checks for missing citations and PII patterns."""
    if require_citations and not response.citations:
        return False, "Response lacks source citations."

    if _PII_PATTERN.search(response.answer):
        return False, "Response may contain PII (SSN or card number detected)."

    return True, "OK"
