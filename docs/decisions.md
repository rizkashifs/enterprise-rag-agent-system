# Architecture Decisions

## ADR-001: Retrieval and generation are separate layers

Retrieval quality, citation policy, prompt construction, and answer generation should be independently measurable.

## ADR-002: Agents require constrained tools

Agent orchestration should expose tools through explicit contracts, permissions, rate limits, and audit logs.
