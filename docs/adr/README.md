# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the Synthetic Data Guardian project. ADRs document important architectural decisions, their context, rationale, and consequences.

## ADR Format

We use the standard ADR format with the following sections:
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: The situation that motivates the decision
- **Decision**: The chosen solution
- **Rationale**: Why this solution was selected
- **Consequences**: Positive and negative outcomes
- **Implementation**: How the decision is implemented
- **Review**: When and how the decision will be reviewed

## Current ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-architecture-decisions.md) | Architecture Decisions | Accepted | 2024-01-15 |
| [0002](0002-security-and-privacy.md) | Security and Privacy | Accepted | 2024-01-15 |
| [0003](0003-sdlc-checkpointed-implementation.md) | Checkpointed SDLC Implementation Strategy | Accepted | 2024-01-18 |

## Creating New ADRs

1. Copy the template from `docs/adr/template.md`
2. Number sequentially (next available number)
3. Use descriptive title with kebab-case
4. Follow the standard format
5. Update this README with the new entry

## Review Process

ADRs should be reviewed:
- When implementation is complete
- When assumptions change
- Annually for strategic decisions
- When superseded by new requirements