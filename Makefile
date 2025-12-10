# Makefile for RL GridWorld project
# Simplifies common development tasks

.PHONY: help install install-dev test test-verbose test-coverage \
        clean clean-pyc clean-test lint format docs run-demo \
        run-examples compare-algorithms visualize setup-dirs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)RL GridWorld - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Installation targets
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete!$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy
	@echo "$(GREEN)✓ Development environment ready!$(NC)"

install-all: ## Install all dependencies (production + dev + extras)
	@echo "$(BLUE)Installing all dependencies...$(NC)"
	pip install -e ".[all]"
	@echo "$(GREEN)✓ All dependencies installed!$(NC)"

# Testing targets
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v

test-verbose: ## Run tests with detailed output
	@echo "$(BLUE)Running tests with verbose output...$(NC)"
	pytest tests/ -vv --tb=long

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ --cov=. --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/index.html$(NC)"

test-fast: ## Run only fast tests (skip slow ones)
	@echo "$(BLUE)Running fast tests...$(NC)"
	pytest tests/ -v -m "not slow"

test-env: ## Run environment tests only
	@echo "$(BLUE)Running environment tests...$(NC)"
	pytest tests/test_environment.py -v

test-algo: ## Run algorithm tests only
	@echo "$(BLUE)Running algorithm tests...$(NC)"
	pytest tests/test_algorithms.py -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/test_integration.py -v

# Code quality targets
lint: ## Run linting checks
	@echo "$(BLUE)Running flake8...$(NC)"
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "$(GREEN)✓ Linting complete!$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code with black...$(NC)"
	black .
	@echo "$(GREEN)✓ Code formatted!$(NC)"

format-check: ## Check code formatting without modifying
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check .

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy . --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking complete!$(NC)"

# Cleaning targets
clean: clean-pyc clean-test clean-build ## Remove all build, test, and Python artifacts

clean-pyc: ## Remove Python file artifacts
	@echo "$(BLUE)Cleaning Python artifacts...$(NC)"
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	@echo "$(GREEN)✓ Python artifacts cleaned!$(NC)"

clean-test: ## Remove test and coverage artifacts
	@echo "$(BLUE)Cleaning test artifacts...$(NC)"
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	@echo "$(GREEN)✓ Test artifacts cleaned!$(NC)"

clean-build: ## Remove build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	@echo "$(GREEN)✓ Build artifacts cleaned!$(NC)"

# Setup targets
setup-dirs: ## Create necessary directories
	@echo "$(BLUE)Creating directories...$(NC)"
	mkdir -p results logs saved_models plots
	@echo "$(GREEN)✓ Directories created!$(NC)"

# Running targets
run-demo: ## Run value iteration demo
	@echo "$(BLUE)Running Value Iteration demo...$(NC)"
	python value_agent.py

run-policy: ## Run policy iteration demo
	@echo "$(BLUE)Running Policy Iteration demo...$(NC)"
	python policy_iteration.py

run-random: ## Run random agent demo
	@echo "$(BLUE)Running Random Agent demo...$(NC)"
	python random_agent.py

run-qlearning-moving: ## Run Q-Learning with moving goal
	@echo "$(BLUE)Running Q-Learning (Moving Goal)...$(NC)"
	python q_learning_moving_goal.py

run-qlearning-dynamic: ## Run Q-Learning with dynamic goal
	@echo "$(BLUE)Running Q-Learning (Dynamic Goal)...$(NC)"
	python q_learning_dynamic_goal.py

run-examples: ## Run advanced examples
	@echo "$(BLUE)Running advanced examples...$(NC)"
	python examples/advanced_usage.py

# Analysis targets
compare-algorithms: ## Compare all algorithms
	@echo "$(BLUE)Running algorithm comparison...$(NC)"
	python performance_comparison.py

visualize: ## Generate visualizations
	@echo "$(BLUE)Generating visualizations...$(NC)"
	python visualization_utils.py

# Documentation targets
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)✓ Documentation generated in docs/_build/html/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	cd docs/_build/html && python -m http.server 8000

# Development workflow targets
dev-setup: install-dev setup-dirs ## Complete development setup
	@echo "$(GREEN)✓ Development environment is ready!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Run 'make test' to verify everything works"
	@echo "  2. Run 'make run-demo' to see a demo"
	@echo "  3. Run 'make help' to see all available commands"

check: lint test ## Run all checks (lint + tests)
	@echo "$(GREEN)✓ All checks passed!$(NC)"

ci: lint test-coverage ## Run CI checks
	@echo "$(GREEN)✓ CI checks passed!$(NC)"

# Benchmarking targets
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	@echo "$(YELLOW)Small grid (5x5):$(NC)"
	time python value_agent.py
	@echo ""
	@echo "$(YELLOW)Medium grid (7x7):$(NC)"
	time python policy_iteration.py

# Quick start targets
quickstart: install setup-dirs run-demo ## Quick start (install + demo)
	@echo "$(GREEN)✓ Quick start complete!$(NC)"

# Release targets
build: clean ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(NC)"
	python setup.py sdist bdist_wheel
	@echo "$(GREEN)✓ Packages built in dist/$(NC)"

release-test: build ## Upload to TestPyPI
	@echo "$(BLUE)Uploading to TestPyPI...$(NC)"
	twine upload --repository testpypi dist/*

release: build ## Upload to PyPI
	@echo "$(RED)WARNING: This will upload to PyPI!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		twine upload dist/*; \
		echo "$(GREEN)✓ Released to PyPI!$(NC)"; \
	fi

# Info targets
info: ## Display project information
	@echo "$(BLUE)RL GridWorld Project Information$(NC)"
	@echo ""
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo ""
	@echo "Project structure:"
	@tree -L 2 -I '__pycache__|*.pyc|htmlcov|.pytest_cache'

status: ## Show git status and recent changes
	@echo "$(BLUE)Git Status:$(NC)"
	@git status -s
	@echo ""
	@echo "$(BLUE)Recent Commits:$(NC)"
	@git log --oneline -5

# Combined targets for common workflows
full-test: clean test-coverage lint ## Clean, test with coverage, and lint
	@echo "$(GREEN)✓ Full test suite complete!$(NC)"

pre-commit: format lint test ## Run pre-commit checks
	@echo "$(GREEN)✓ Pre-commit checks passed!$(NC)"

pre-push: clean format lint test-coverage ## Run pre-push checks
	@echo "$(GREEN)✓ Pre-push checks passed!$(NC)"