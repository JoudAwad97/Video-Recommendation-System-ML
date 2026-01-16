# =============================================================================
# Video Recommendation System - Makefile
# =============================================================================
# Common development and deployment tasks
#
# Usage:
#   make help              - Show available commands
#   make install           - Install dependencies
#   make test              - Run tests
#   make deploy-dev        - Deploy to dev environment
# =============================================================================

.PHONY: help install install-dev test test-cov lint format clean \
        docker-build docker-run deploy-dev deploy-staging deploy-prod \
        bootstrap-data train-models local-serve docs

# Default target
.DEFAULT_GOAL := help

# Environment
PYTHON := python3
PIP := pip3
PYTEST := pytest
CDK := npx cdk

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "$(CYAN)Video Recommendation System$(RESET)"
	@echo ""
	@echo "$(GREEN)Usage:$(RESET) make [target]"
	@echo ""
	@echo "$(GREEN)Available targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================

install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	pre-commit install

install-cdk: ## Install CDK dependencies
	cd deployment/cdk && npm install

# =============================================================================
# Testing
# =============================================================================

test: ## Run tests
	$(PYTEST) tests/ -v

test-cov: ## Run tests with coverage
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-unit: ## Run unit tests only
	$(PYTEST) tests/ -v -m "not integration"

test-integration: ## Run integration tests only
	$(PYTEST) tests/ -v -m integration

test-fast: ## Run tests in parallel (requires pytest-xdist)
	$(PYTEST) tests/ -v -n auto

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linters (ruff, mypy)
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format: ## Format code with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

check: lint test ## Run all checks (lint + tests)

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Lambda Docker image
	docker build -f deployment/docker/Dockerfile.lambda -t video-rec-lambda .

docker-build-arm: ## Build Lambda Docker image for ARM64
	docker buildx build --platform linux/arm64 \
		-f deployment/docker/Dockerfile.lambda \
		-t video-rec-lambda:arm64 .

docker-run: ## Run Lambda locally
	docker run -p 8080:8080 \
		-e ENVIRONMENT=local \
		-e LOG_FORMAT=text \
		video-rec-lambda

docker-shell: ## Shell into Lambda container
	docker run -it --entrypoint /bin/bash video-rec-lambda

# =============================================================================
# Local Development
# =============================================================================

local-serve: ## Run local recommendation server
	$(PYTHON) scripts/serve_recommendations.py --local

local-test: ## Test recommendations locally
	curl -X POST http://localhost:8080/recommendations \
		-H "Content-Type: application/json" \
		-d '{"user_id": 1, "num_recommendations": 10}'

bootstrap-data: ## Generate and populate synthetic data
	$(PYTHON) scripts/bootstrap_data.py --num-users 1000 --num-videos 500

bootstrap-data-local: ## Generate synthetic data (local mode, no AWS)
	$(PYTHON) scripts/bootstrap_data.py --local --num-users 100 --num-videos 50

# =============================================================================
# Training
# =============================================================================

train-two-tower: ## Train Two-Tower model
	$(PYTHON) scripts/train_two_tower.py

train-ranker: ## Train Ranker model
	$(PYTHON) scripts/train_ranker.py

train-all: train-two-tower train-ranker ## Train all models

process-data: ## Run data processing pipeline
	$(PYTHON) scripts/run_processing.py

# =============================================================================
# Deployment
# =============================================================================

deploy-dev: ## Deploy to dev environment
	cd deployment/cdk && $(CDK) deploy --all -c environment=dev --require-approval never

deploy-staging: ## Deploy to staging environment
	cd deployment/cdk && $(CDK) deploy --all -c environment=staging

deploy-prod: ## Deploy to production environment
	@echo "$(YELLOW)WARNING: Deploying to PRODUCTION$(RESET)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	cd deployment/cdk && $(CDK) deploy --all -c environment=prod

deploy-diff: ## Show deployment diff
	cd deployment/cdk && $(CDK) diff --all

deploy-synth: ## Synthesize CloudFormation templates
	cd deployment/cdk && $(CDK) synth

destroy-dev: ## Destroy dev environment (DANGEROUS)
	@echo "$(YELLOW)WARNING: Destroying dev environment$(RESET)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	cd deployment/cdk && $(CDK) destroy --all -c environment=dev

# =============================================================================
# ML Pipeline
# =============================================================================

trigger-pipeline: ## Trigger ML pipeline (Step Functions)
	aws stepfunctions start-execution \
		--state-machine-arn $$(aws stepfunctions list-state-machines --query "stateMachines[?contains(name, 'ml-pipeline')].stateMachineArn | [0]" --output text) \
		--input '{"trigger": "manual"}'

pipeline-status: ## Check ML pipeline status
	aws stepfunctions list-executions \
		--state-machine-arn $$(aws stepfunctions list-state-machines --query "stateMachines[?contains(name, 'ml-pipeline')].stateMachineArn | [0]" --output text) \
		--max-results 5

# =============================================================================
# Utilities
# =============================================================================

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean ## Clean everything including dependencies
	rm -rf .venv/
	rm -rf node_modules/
	rm -rf deployment/cdk/node_modules/
	rm -rf deployment/cdk/cdk.out/

logs: ## Tail Lambda logs
	aws logs tail /aws/lambda/VideoRecSystem-dev-recommendation --follow

logs-ml: ## Tail ML pipeline logs
	aws logs tail /aws/stepfunctions/VideoRecSystem-dev-ml-pipeline --follow

# =============================================================================
# Documentation
# =============================================================================

docs: ## Generate documentation
	@echo "Documentation generation not yet configured"

api-docs: ## Generate API documentation (requires running server)
	@echo "Fetching OpenAPI spec from running server..."
	curl -s http://localhost:8080/openapi.json > docs/openapi.json

# =============================================================================
# Environment Setup
# =============================================================================

setup-venv: ## Create virtual environment
	$(PYTHON) -m venv .venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

setup-env: ## Copy environment template
	cp -n .env.example .env || true
	@echo "Edit .env with your configuration"

setup: setup-venv setup-env install-dev install-cdk ## Complete development setup
	@echo "$(GREEN)Setup complete!$(RESET)"
	@echo "Activate virtual environment: source .venv/bin/activate"
