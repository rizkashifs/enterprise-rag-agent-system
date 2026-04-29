from src.core.models import PolicyDecision

FORBIDDEN_TOPICS = {"password", "credentials", "secret", "hack", "exploit", "bypass"}


class PolicyLayer:
    """Simple keyword-based allow/deny policy for incoming requests."""

    def check(self, request: str) -> PolicyDecision:
        lower = request.lower()
        for topic in FORBIDDEN_TOPICS:
            if topic in lower:
                return PolicyDecision(
                    allowed=False,
                    reason=f"Request contains forbidden topic: '{topic}'",
                )
        return PolicyDecision(allowed=True, reason="Approved")
