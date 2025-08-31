# CTDS Makefile
# Convenient commands for development and testing

# Python executable - use virtual environment if available
PYTHON := $(shell if [ -d ".venv" ]; then echo ".venv/bin/python"; else echo "python"; fi)
PIP := $(shell if [ -d ".venv" ]; then echo ".venv/bin/pip"; else echo "pip"; fi)

.PHONY: help setup install clean test docs demo format lint type-check all

# Default target
help:
	@echo "CTDS Development Commands"
	@echo "========================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup     - Run full setup (same as ./setup.sh)"
	@echo "  make install   - Install package in development mode"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test      - Run test suite"
	@echo "  make demo      - Run quickstart demo"
	@echo "  make format    - Format code with black"
	@echo "  make lint      - Lint code with ruff"
	@echo "  make type-check - Type check with mypy"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs      - Build and serve documentation"
	@echo "  make docs-build - Build documentation only"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean     - Clean build artifacts"
	@echo "  make clean-all - Clean everything including venv"

# Setup and installation
setup:
	@echo "🔧 Running full setup..."
	./setup.sh

install:
	@echo "📦 Installing CTDS in development mode..."
	$(PIP) install -e ".[dev,docs,examples]"

# Testing and demos


demo:
	@echo "🚀 Running quickstart demo..."
	$(PYTHON) quickstart.py

# Code quality
format:
	@echo "🎨 Formatting code with black..."
	$(PYTHON) -m black src/ tests/ --line-length 88

lint:
	@echo "🔍 Linting code with ruff..."
	$(PYTHON) -m ruff check src/ tests/

type-check:
	@echo "🔎 Type checking with mypy..."
	$(PYTHON) -m mypy src/ctds/

# Documentation
docs:
	@echo "📚 Building and serving documentation..."
	mkdocs serve

docs-build:
	@echo "📖 Building documentation..."
	mkdocs build

# Cleanup
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete

clean-all: clean
	@echo "🧹 Cleaning everything including virtual environment..."
	rm -rf .venv/

# All quality checks
all: format lint type-check test
	@echo "✅ All quality checks passed!"
