# =============================================================================
# Synthetic Data Guardian - Makefile
# =============================================================================

.PHONY: help install build test lint format clean docker deploy monitor

# Default target
.DEFAULT_GOAL := help

# Variables
APP_NAME := synthetic-data-guardian
VERSION := $(shell grep -o '"version": "[^"]*"' package.json | grep -o '[0-9][^"]*')
DOCKER_REGISTRY := ghcr.io/your-org
DOCKER_IMAGE := $(DOCKER_REGISTRY)/$(APP_NAME)
NAMESPACE := synthetic-guardian

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Help target
help: ## Display this help message
	@echo "$(CYAN)Synthetic Data Guardian - Available Commands$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development targets
install: ## Install dependencies
	@echo "$(CYAN)Installing dependencies...$(RESET)"
	npm install
	pip install -r requirements-dev.txt
	pip install -e ".[dev]"
	npx husky install
	pre-commit install

install-python: ## Install Python dependencies only
	@echo "$(CYAN)Installing Python dependencies...$(RESET)"
	pip install -r requirements-dev.txt
	pip install -e ".[dev]"
	pre-commit install

install-node: ## Install Node.js dependencies only
	@echo "$(CYAN)Installing Node.js dependencies...$(RESET)"
	npm install
	npx husky install

dev: ## Start development environment
	@echo "$(CYAN)Starting development environment...$(RESET)"
	docker-compose -f docker-compose.dev.yml up -d

dev-stop: ## Stop development environment
	@echo "$(CYAN)Stopping development environment...$(RESET)"
	docker-compose -f docker-compose.dev.yml down

dev-logs: ## View development logs
	@echo "$(CYAN)Viewing development logs...$(RESET)"
	docker-compose -f docker-compose.dev.yml logs -f

# Build targets
build: ## Build the application
	@echo "$(CYAN)Building application...$(RESET)"
	npm run build

build-docker: ## Build Docker image
	@echo "$(CYAN)Building Docker image...$(RESET)"
	docker build -t $(APP_NAME):$(VERSION) .
	docker tag $(APP_NAME):$(VERSION) $(APP_NAME):latest

build-docker-dev: ## Build development Docker image
	@echo "$(CYAN)Building development Docker image...$(RESET)"
	docker build -f Dockerfile.dev -t $(APP_NAME):dev .

# Testing targets
test: ## Run all tests
	@echo "$(CYAN)Running tests...$(RESET)"
	npm run test
	pytest

test-python: ## Run Python tests only
	@echo "$(CYAN)Running Python tests...$(RESET)"
	pytest

test-node: ## Run Node.js tests only
	@echo "$(CYAN)Running Node.js tests...$(RESET)"
	npm run test

test-unit: ## Run unit tests only
	@echo "$(CYAN)Running unit tests...$(RESET)"
	npm run test -- --testPathPattern="src/"

test-integration: ## Run integration tests
	@echo "$(CYAN)Running integration tests...$(RESET)"
	npm run test:integration

test-e2e: ## Run end-to-end tests
	@echo "$(CYAN)Running end-to-end tests...$(RESET)"
	npm run test:e2e

test-performance: ## Run performance tests
	@echo "$(CYAN)Running performance tests...$(RESET)"
	npm run test:performance

test-coverage: ## Generate test coverage report
	@echo "$(CYAN)Generating coverage report...$(RESET)"
	npm run test:coverage

# Code quality targets
lint: ## Run linting
	@echo "$(CYAN)Running linter...$(RESET)"
	npm run lint
	ruff check src/ tests/
	flake8 src/ tests/

lint-fix: ## Fix linting issues
	@echo "$(CYAN)Fixing linting issues...$(RESET)"
	npm run lint:fix
	ruff check --fix src/ tests/
	black src/ tests/
	isort src/ tests/

lint-python: ## Run Python linting only
	@echo "$(CYAN)Running Python linting...$(RESET)"
	ruff check src/ tests/
	flake8 src/ tests/
	mypy src/

lint-node: ## Run Node.js linting only
	@echo "$(CYAN)Running Node.js linting...$(RESET)"
	npm run lint

format: ## Format code
	@echo "$(CYAN)Formatting code...$(RESET)"
	npm run format
	black src/ tests/
	isort src/ tests/

format-python: ## Format Python code only
	@echo "$(CYAN)Formatting Python code...$(RESET)"
	black src/ tests/
	isort src/ tests/

format-node: ## Format Node.js code only
	@echo "$(CYAN)Formatting Node.js code...$(RESET)"
	npm run format

format-check: ## Check code formatting
	@echo "$(CYAN)Checking code formatting...$(RESET)"
	npm run format:check
	black --check src/ tests/
	isort --check-only src/ tests/

typecheck: ## Run type checking
	@echo "$(CYAN)Running type checking...$(RESET)"
	npm run typecheck
	mypy src/

# Security targets
security-audit: ## Run security audit
	@echo "$(CYAN)Running security audit...$(RESET)"
	npm run security:audit

security-scan: ## Run security scan
	@echo "$(CYAN)Running security scan...$(RESET)"
	npm run security:scan

# Cleanup targets
clean: ## Clean build artifacts
	@echo "$(CYAN)Cleaning build artifacts...$(RESET)"
	npm run clean
	rm -rf node_modules/.cache
	docker system prune -f

clean-all: ## Clean everything including dependencies
	@echo "$(CYAN)Cleaning everything...$(RESET)"
	rm -rf node_modules
	rm -rf dist
	rm -rf coverage
	rm -rf .nyc_output
	rm -rf test-results
	docker system prune -af

# Docker targets
docker-run: ## Run Docker container
	@echo "$(CYAN)Running Docker container...$(RESET)"
	docker run -d --name $(APP_NAME) -p 8080:8080 $(APP_NAME):latest

docker-stop: ## Stop Docker container
	@echo "$(CYAN)Stopping Docker container...$(RESET)"
	docker stop $(APP_NAME) && docker rm $(APP_NAME)

docker-push: ## Push Docker image to registry
	@echo "$(CYAN)Pushing Docker image to registry...$(RESET)"
	docker tag $(APP_NAME):$(VERSION) $(DOCKER_IMAGE):$(VERSION)
	docker tag $(APP_NAME):$(VERSION) $(DOCKER_IMAGE):latest
	docker push $(DOCKER_IMAGE):$(VERSION)
	docker push $(DOCKER_IMAGE):latest

docker-compose-up: ## Start production environment with docker-compose
	@echo "$(CYAN)Starting production environment...$(RESET)"
	docker-compose up -d

docker-compose-down: ## Stop production environment
	@echo "$(CYAN)Stopping production environment...$(RESET)"
	docker-compose down

docker-compose-logs: ## View production logs
	@echo "$(CYAN)Viewing production logs...$(RESET)"
	docker-compose logs -f

# Database targets
db-migrate: ## Run database migrations
	@echo "$(CYAN)Running database migrations...$(RESET)"
	npm run db:migrate

db-seed: ## Seed database with test data
	@echo "$(CYAN)Seeding database...$(RESET)"
	npm run db:seed

db-reset: ## Reset database
	@echo "$(CYAN)Resetting database...$(RESET)"
	npm run db:reset

# Deployment targets
deploy-staging: ## Deploy to staging environment
	@echo "$(CYAN)Deploying to staging...$(RESET)"
	kubectl apply -f k8s/staging/ -n $(NAMESPACE)-staging

deploy-production: ## Deploy to production environment
	@echo "$(CYAN)Deploying to production...$(RESET)"
	kubectl apply -f k8s/production/ -n $(NAMESPACE)

deploy-rollback: ## Rollback deployment
	@echo "$(CYAN)Rolling back deployment...$(RESET)"
	kubectl rollout undo deployment/$(APP_NAME) -n $(NAMESPACE)

# Monitoring targets
monitor: ## Start monitoring stack
	@echo "$(CYAN)Starting monitoring stack...$(RESET)"
	docker-compose -f docker-compose.monitoring.yml up -d

monitor-stop: ## Stop monitoring stack
	@echo "$(CYAN)Stopping monitoring stack...$(RESET)"
	docker-compose -f docker-compose.monitoring.yml down

logs: ## View application logs
	@echo "$(CYAN)Viewing application logs...$(RESET)"
	kubectl logs -f deployment/$(APP_NAME) -n $(NAMESPACE)

# Documentation targets
docs-build: ## Build documentation
	@echo "$(CYAN)Building documentation...$(RESET)"
	npm run docs:build

docs-serve: ## Serve documentation locally
	@echo "$(CYAN)Serving documentation...$(RESET)"
	npm run docs:serve

# Release targets
release: ## Create a new release
	@echo "$(CYAN)Creating new release...$(RESET)"
	npm run semantic-release

version: ## Show current version
	@echo "$(CYAN)Current version:$(RESET) $(VERSION)"

# Utility targets
check-deps: ## Check for outdated dependencies
	@echo "$(CYAN)Checking dependencies...$(RESET)"
	npm outdated
	pip list --outdated

update-deps: ## Update dependencies
	@echo "$(CYAN)Updating dependencies...$(RESET)"
	npm update
	pip install -U -r requirements-dev.txt

health-check: ## Perform health check
	@echo "$(CYAN)Performing health check...$(RESET)"
	curl -f http://localhost:8080/api/v1/health || echo "$(RED)Health check failed$(RESET)"

# CI/CD targets
ci-install: ## Install dependencies for CI
	@echo "$(CYAN)Installing CI dependencies...$(RESET)"
	npm ci --silent

ci-test: ## Run tests in CI environment
	@echo "$(CYAN)Running CI tests...$(RESET)"
	npm run test:coverage
	npm run test:integration
	npm run lint
	npm run typecheck

ci-build: ## Build for CI
	@echo "$(CYAN)Building for CI...$(RESET)"
	npm run build
	docker build -t $(APP_NAME):ci .

# Performance targets
benchmark: ## Run performance benchmarks
	@echo "$(CYAN)Running benchmarks...$(RESET)"
	npm run test:performance

profile: ## Profile application performance
	@echo "$(CYAN)Profiling application...$(RESET)"
	npm run profile

# Debug targets
debug: ## Start application in debug mode
	@echo "$(CYAN)Starting in debug mode...$(RESET)"
	npm run debug

inspect: ## Inspect running container
	@echo "$(CYAN)Inspecting container...$(RESET)"
	docker exec -it $(APP_NAME) /bin/sh