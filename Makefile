.PHONY: help test test-verbose install clean format lint setup

help:
	@echo "ScalarGrad Development Commands"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Install development dependencies"
	@echo "  make install        Install in development mode"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-verbose   Run tests with verbose output"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format         Format code with black"
	@echo "  make lint           Check code with pylint"
	@echo "  make clean          Remove build artifacts"
	@echo ""
	@echo "Development:"
	@echo "  make docs           Generate documentation"
	@echo "  make examples       Run example notebooks"
	@echo ""

setup:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest black pylint

install:
	@echo "Installing ScalarGrad in development mode..."
	pip install -e .

test:
	@echo "Running all tests..."
	python tests/test_derivatives.py
	python tests/test_nn_full.py
	python tests/test_sub_neg.py
	@echo ""
	@echo "✓ All tests passed!"

test-verbose:
	@echo "Running tests with verbose output..."
	@echo ""
	@echo "=== Derivatives Tests ==="
	python tests/test_derivatives.py
	@echo ""
	@echo "=== Neural Network Tests ==="
	python tests/test_nn_full.py
	@echo ""
	@echo "=== Arithmetic Tests ==="
	python tests/test_sub_neg.py

format:
	@echo "Formatting code with black..."
	black scalargrad/ tests/ --line-length=88

lint:
	@echo "Linting code with pylint..."
	pylint scalargrad/ tests/ --disable=missing-docstring,too-few-public-methods

clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	@echo "✓ Clean complete"

docs:
	@echo "Generating documentation..."
	@echo "Documentation files:"
	@echo "  - README.md"
	@echo "  - ARCHITECTURE.md"
	@echo "  - CONTRIBUTING.md"
	@echo "  - TESTING.md"

examples:
	@echo "Running example scripts..."
	@echo "See test_*.py files for usage examples"

.DEFAULT_GOAL := help
