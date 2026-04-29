from src.core.models import Document

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
