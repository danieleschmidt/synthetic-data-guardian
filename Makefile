# Synthetic Data Guardian Makefile
# Standardized build commands and development workflows

.PHONY: help install install-dev clean test test-unit test-integration test-e2e lint format security build docker-build docker-run docker-stop start stop logs status backup restore deploy docs serve-docs release

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
POETRY := poetry
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := synthetic-data-guardian
IMAGE_NAME := terragon/synthetic-data-guardian
VERSION := $(shell grep '^version' pyproject.toml | cut -d'"' -f2)
BUILD_DATE := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
VCS_REF := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

# Help target
help: ## Show this help message
	@echo "$(BLUE)Synthetic Data Guardian - Development Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Available targets:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Examples:$(RESET)"
	@echo "  make install      # Install dependencies"
	@echo "  make test         # Run all tests"
	@echo "  make docker-build # Build Docker image"
	@echo "  make start        # Start development environment"

# =============================================================================
# DEVELOPMENT SETUP
# =============================================================================

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(RESET)"
	$(POETRY) install --only=main
	@echo "$(GREEN)✓ Production dependencies installed$(RESET)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(POETRY) install --with dev
	$(POETRY) run pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(RESET)"

clean: ## Clean build artifacts and cache
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml
	@echo "$(GREEN)✓ Build artifacts cleaned$(RESET)"

# =============================================================================
# TESTING
# =============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(RESET)"
	$(POETRY) run pytest tests/ -v --cov=src/synthetic_guardian --cov-report=html --cov-report=term-missing --cov-report=xml
	@echo "$(GREEN)✓ All tests completed$(RESET)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(POETRY) run pytest tests/unit/ -v -m "unit"
	@echo "$(GREEN)✓ Unit tests completed$(RESET)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(POETRY) run pytest tests/integration/ -v -m "integration"
	@echo "$(GREEN)✓ Integration tests completed$(RESET)"

test-e2e: ## Run end-to-end tests only
	@echo "$(BLUE)Running end-to-end tests...$(RESET)"
	$(POETRY) run pytest tests/e2e/ -v -m "e2e"
	@echo "$(GREEN)✓ End-to-end tests completed$(RESET)"

test-coverage: ## Generate test coverage report
	@echo "$(BLUE)Generating coverage report...$(RESET)"
	$(POETRY) run pytest tests/ --cov=src/synthetic_guardian --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(RESET)"

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(RESET)"
	$(POETRY) run pytest tests/ -v -m "slow" --durations=10
	@echo "$(GREEN)✓ Performance tests completed$(RESET)"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(RESET)"
	$(POETRY) run ruff check src/ tests/
	$(POETRY) run pylint src/synthetic_guardian/
	$(POETRY) run mypy src/synthetic_guardian/
	@echo "$(GREEN)✓ Linting checks completed$(RESET)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(POETRY) run black src/ tests/
	$(POETRY) run isort src/ tests/
	$(POETRY) run ruff format src/ tests/
	@echo "$(GREEN)✓ Code formatting completed$(RESET)"

format-check: ## Check code formatting
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	$(POETRY) run black --check src/ tests/
	$(POETRY) run isort --check-only src/ tests/
	$(POETRY) run ruff format --check src/ tests/
	@echo "$(GREEN)✓ Code formatting check completed$(RESET)"

typecheck: ## Run type checking
	@echo "$(BLUE)Running type checks...$(RESET)"
	$(POETRY) run mypy src/synthetic_guardian/
	@echo "$(GREEN)✓ Type checking completed$(RESET)"

# =============================================================================
# SECURITY
# =============================================================================

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	$(POETRY) run bandit -r src/synthetic_guardian/ -f json -o bandit-report.json
	$(POETRY) run safety check --json --output safety-report.json
	@echo "$(GREEN)✓ Security checks completed$(RESET)"

security-audit: ## Run comprehensive security audit
	@echo "$(BLUE)Running comprehensive security audit...$(RESET)"
	$(POETRY) run bandit -r src/
	$(POETRY) run safety check
	$(POETRY) run pip-audit
	@echo "$(GREEN)✓ Security audit completed$(RESET)"

# =============================================================================
# BUILDING
# =============================================================================

build: ## Build Python package
	@echo "$(BLUE)Building Python package...$(RESET)"
	$(POETRY) build
	@echo "$(GREEN)✓ Package built in dist/$(RESET)"

build-wheel: ## Build wheel package only
	@echo "$(BLUE)Building wheel package...$(RESET)"
	$(POETRY) build --format wheel
	@echo "$(GREEN)✓ Wheel package built$(RESET)"

build-sdist: ## Build source distribution only
	@echo "$(BLUE)Building source distribution...$(RESET)"
	$(POETRY) build --format sdist
	@echo "$(GREEN)✓ Source distribution built$(RESET)"

# =============================================================================
# DOCKER
# =============================================================================

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	$(DOCKER) build \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg VCS_REF=$(VCS_REF) \
		--build-arg VERSION=$(VERSION) \
		-t $(IMAGE_NAME):$(VERSION) \
		-t $(IMAGE_NAME):latest \
		.
	@echo "$(GREEN)✓ Docker image built: $(IMAGE_NAME):$(VERSION)$(RESET)"

docker-build-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(RESET)"
	$(DOCKER) build \
		--target development \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg VCS_REF=$(VCS_REF) \
		--build-arg VERSION=$(VERSION) \
		-t $(IMAGE_NAME):dev \
		.
	@echo "$(GREEN)✓ Development Docker image built$(RESET)"

docker-build-prod: ## Build production Docker image
	@echo "$(BLUE)Building production Docker image...$(RESET)"
	$(DOCKER) build \
		--target production \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg VCS_REF=$(VCS_REF) \
		--build-arg VERSION=$(VERSION) \
		-t $(IMAGE_NAME):$(VERSION) \
		-t $(IMAGE_NAME):latest \
		.
	@echo "$(GREEN)✓ Production Docker image built$(RESET)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(RESET)"
	$(DOCKER) run -d \
		--name $(PROJECT_NAME) \
		-p 8080:8080 \
		--env-file .env \
		$(IMAGE_NAME):latest
	@echo "$(GREEN)✓ Docker container started$(RESET)"

docker-stop: ## Stop Docker container
	@echo "$(BLUE)Stopping Docker container...$(RESET)"
	$(DOCKER) stop $(PROJECT_NAME) || true
	$(DOCKER) rm $(PROJECT_NAME) || true
	@echo "$(GREEN)✓ Docker container stopped$(RESET)"

docker-logs: ## Show Docker container logs
	$(DOCKER) logs -f $(PROJECT_NAME)

docker-shell: ## Open shell in Docker container
	$(DOCKER) exec -it $(PROJECT_NAME) /bin/bash

# =============================================================================
# DEVELOPMENT ENVIRONMENT
# =============================================================================

start: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(RESET)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Development environment started$(RESET)"
	@echo "$(YELLOW)Services available at:$(RESET)"
	@echo "  API Server:      http://localhost:8080"
	@echo "  Grafana:         http://localhost:3000 (admin/admin)"
	@echo "  Prometheus:      http://localhost:9090"
	@echo "  Jaeger UI:       http://localhost:16686"
	@echo "  Adminer:         http://localhost:8081"
	@echo "  Redis Commander: http://localhost:8082"

stop: ## Stop development environment
	@echo "$(BLUE)Stopping development environment...$(RESET)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Development environment stopped$(RESET)"

restart: stop start ## Restart development environment

logs: ## Show development environment logs
	$(DOCKER_COMPOSE) logs -f

status: ## Show development environment status
	@echo "$(BLUE)Development environment status:$(RESET)"
	$(DOCKER_COMPOSE) ps

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(RESET)"
	$(POETRY) run alembic upgrade head
	@echo "$(GREEN)✓ Database migrations completed$(RESET)"

db-migrate-create: ## Create new database migration
	@echo "$(BLUE)Creating new database migration...$(RESET)"
	@read -p "Migration message: " msg; \
	$(POETRY) run alembic revision --autogenerate -m "$$msg"
	@echo "$(GREEN)✓ Migration created$(RESET)"

db-reset: ## Reset database (WARNING: destroys data)
	@echo "$(RED)WARNING: This will destroy all data!$(RESET)"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	$(DOCKER_COMPOSE) down -v
	$(DOCKER_COMPOSE) up -d postgres redis neo4j
	$(POETRY) run alembic upgrade head
	@echo "$(GREEN)✓ Database reset completed$(RESET)"

backup: ## Backup databases
	@echo "$(BLUE)Creating database backup...$(RESET)"
	mkdir -p backups
	$(DOCKER_COMPOSE) exec postgres pg_dump -U postgres synthetic_guardian > backups/postgres_$(shell date +%Y%m%d_%H%M%S).sql
	$(DOCKER_COMPOSE) exec redis redis-cli BGSAVE
	@echo "$(GREEN)✓ Database backup completed$(RESET)"

restore: ## Restore database from backup
	@echo "$(BLUE)Restoring database from backup...$(RESET)"
	@ls -la backups/*.sql
	@read -p "Enter backup file name: " file; \
	$(DOCKER_COMPOSE) exec -T postgres psql -U postgres synthetic_guardian < "$$file"
	@echo "$(GREEN)✓ Database restore completed$(RESET)"

# =============================================================================
# MONITORING AND DEBUGGING
# =============================================================================

health: ## Check application health
	@echo "$(BLUE)Checking application health...$(RESET)"
	curl -f http://localhost:8080/health || echo "$(RED)Health check failed$(RESET)"
	curl -f http://localhost:8080/ready || echo "$(RED)Readiness check failed$(RESET)"

metrics: ## Show application metrics
	@echo "$(BLUE)Application metrics:$(RESET)"
	curl -s http://localhost:8080/metrics | head -20

profile: ## Run performance profiling
	@echo "$(BLUE)Running performance profiling...$(RESET)"
	$(POETRY) run py-spy record -o profile.svg -d 30 -- python -m synthetic_guardian.api.main
	@echo "$(GREEN)✓ Profile saved as profile.svg$(RESET)"

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	$(POETRY) run mkdocs build
	@echo "$(GREEN)✓ Documentation built in site/$(RESET)"

serve-docs: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(RESET)"
	$(POETRY) run mkdocs serve

docs-deploy: ## Deploy documentation
	@echo "$(BLUE)Deploying documentation...$(RESET)"
	$(POETRY) run mkdocs gh-deploy
	@echo "$(GREEN)✓ Documentation deployed$(RESET)"

# =============================================================================
# RELEASE MANAGEMENT
# =============================================================================

version: ## Show current version
	@echo "Current version: $(VERSION)"

version-bump-patch: ## Bump patch version
	@echo "$(BLUE)Bumping patch version...$(RESET)"
	$(POETRY) version patch
	@echo "$(GREEN)✓ Version bumped to $(shell $(POETRY) version -s)$(RESET)"

version-bump-minor: ## Bump minor version
	@echo "$(BLUE)Bumping minor version...$(RESET)"
	$(POETRY) version minor
	@echo "$(GREEN)✓ Version bumped to $(shell $(POETRY) version -s)$(RESET)"

version-bump-major: ## Bump major version
	@echo "$(BLUE)Bumping major version...$(RESET)"
	$(POETRY) version major
	@echo "$(GREEN)✓ Version bumped to $(shell $(POETRY) version -s)$(RESET)"

release: ## Create a new release
	@echo "$(BLUE)Creating new release...$(RESET)"
	@echo "Current version: $(VERSION)"
	@read -p "Enter new version (or press Enter to auto-bump patch): " new_version; \
	if [ -n "$$new_version" ]; then \
		$(POETRY) version "$$new_version"; \
	else \
		$(POETRY) version patch; \
	fi
	@$(MAKE) clean test lint security build
	git add pyproject.toml
	git commit -m "chore: bump version to $(shell $(POETRY) version -s)"
	git tag -a "v$(shell $(POETRY) version -s)" -m "Release v$(shell $(POETRY) version -s)"
	@echo "$(GREEN)✓ Release v$(shell $(POETRY) version -s) created$(RESET)"
	@echo "$(YELLOW)Don't forget to: git push && git push --tags$(RESET)"

# =============================================================================
# DEPLOYMENT
# =============================================================================

deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(RESET)"
	@echo "$(YELLOW)This would deploy to staging environment$(RESET)"
	# Add your staging deployment commands here

deploy-production: ## Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(RESET)"
	@echo "$(RED)WARNING: Production deployment!$(RESET)"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	@echo "$(YELLOW)This would deploy to production environment$(RESET)"
	# Add your production deployment commands here

# =============================================================================
# UTILITIES
# =============================================================================

check-dependencies: ## Check for outdated dependencies
	@echo "$(BLUE)Checking for outdated dependencies...$(RESET)"
	$(POETRY) show --outdated

update-dependencies: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(POETRY) update
	@echo "$(GREEN)✓ Dependencies updated$(RESET)"

pre-commit: ## Run pre-commit hooks manually
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	$(POETRY) run pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit hooks completed$(RESET)"

install-hooks: ## Install git hooks
	@echo "$(BLUE)Installing git hooks...$(RESET)"
	$(POETRY) run pre-commit install
	@echo "$(GREEN)✓ Git hooks installed$(RESET)"

# =============================================================================
# CI/CD SIMULATION
# =============================================================================

ci: ## Simulate CI pipeline
	@echo "$(BLUE)Simulating CI pipeline...$(RESET)"
	@$(MAKE) clean
	@$(MAKE) install-dev
	@$(MAKE) lint
	@$(MAKE) typecheck
	@$(MAKE) security
	@$(MAKE) test
	@$(MAKE) build
	@echo "$(GREEN)✓ CI pipeline simulation completed$(RESET)"

ci-full: ## Full CI pipeline with Docker
	@echo "$(BLUE)Running full CI pipeline...$(RESET)"
	@$(MAKE) ci
	@$(MAKE) docker-build
	@$(MAKE) docker-build-prod
	@echo "$(GREEN)✓ Full CI pipeline completed$(RESET)"