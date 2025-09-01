.PHONY: help setup install sync clean run run-convert run-stage1 run-stage2 test lint format check dev-install

# Default target
help:
	@echo "Interview Analysis System - Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  setup          - Install uv, sync dependencies, and copy .env template"
	@echo "  install        - Install uv package manager"
	@echo "  sync           - Sync dependencies with uv"
	@echo ""
	@echo "Run Analysis:"
	@echo "  run            - Run complete analysis pipeline"
	@echo "  run-convert    - Convert documents only"
	@echo "  run-stage1     - Run Stage 1 analysis only"
	@echo "  run-stage2     - Run Stage 2 analysis only"
	@echo ""
	@echo "Development:"
	@echo "  dev-install    - Install with development dependencies"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linting (ruff)"
	@echo "  format         - Format code (black + ruff)"
	@echo "  check          - Run type checking (mypy)"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          - Clean build artifacts and cache"

# Setup commands
setup: install sync copy-env

install:
	@echo "Installing uv..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "✓ uv installed successfully"

sync:
	@echo "Syncing dependencies..."
	@uv sync
	@echo "✓ Dependencies synced"

copy-env:
	@if [ ! -f .env ]; then \
		cp .env.template .env; \
		echo "✓ .env template copied - please edit .env with your OpenAI API key"; \
	else \
		echo "✓ .env already exists"; \
	fi

dev-install:
	@echo "Installing with development dependencies..."
	@uv sync --extra dev
	@echo "✓ Development dependencies installed"

# Run analysis commands
run:
	@echo "Running complete analysis pipeline..."
	@uv run src/main_analyzer.py

run-convert:
	@echo "Running document conversion only..."
	@uv run src/main_analyzer.py --stage convert

run-stage1:
	@echo "Running Stage 1 analysis..."
	@uv run src/main_analyzer.py --stage stage1

run-stage2:
	@echo "Running Stage 2 analysis..."
	@uv run src/main_analyzer.py --stage stage2

# Development commands
test:
	@echo "Running tests..."
	@uv run pytest

lint:
	@echo "Running linter..."
	@uv run ruff check src/

format:
	@echo "Formatting code..."
	@uv run black src/
	@uv run ruff check --fix src/

check:
	@echo "Running type checking..."
	@uv run mypy src/

# Cleanup
clean:
	@echo "Cleaning up..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@find . -type d -name __pycache__ -delete
	@find . -type f -name "*.pyc" -delete
	@echo "✓ Cleanup complete"

# Check environment
check-env:
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found. Run 'make copy-env' and edit with your API key"; \
		exit 1; \
	fi
	@if ! grep -q "OPENAI_API_KEY=" .env || grep -q "your_openai_api_key_here" .env; then \
		echo "❌ Please set your OPENAI_API_KEY in .env file"; \
		exit 1; \
	fi
	@echo "✓ Environment configured"