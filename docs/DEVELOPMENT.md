# 🚀 Development Guide

Quick reference for development setup and workflow.

## Prerequisites

- Node.js 18+ and npm 8+
- Python 3.11+
- Docker & Docker Compose
- Git

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd synthetic-data-guardian
cp .env.example .env
npm install && pip install -r requirements-dev.txt

# Start development
make dev  # or docker-compose -f docker-compose.dev.yml up -d
npm run dev
```

## Essential Commands

```bash
npm test              # Run all tests
npm run lint         # Code linting
npm run typecheck    # Type checking
npm run build        # Production build
```

## Documentation

- [Contributing Guide](../CONTRIBUTING.md) - Detailed contribution process
- [Getting Started](guides/getting-started.md) - User getting started guide
- [Architecture](../ARCHITECTURE.md) - System architecture overview

For comprehensive development information, see [CONTRIBUTING.md](../CONTRIBUTING.md).