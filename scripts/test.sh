#!/usr/bin/env bash
# this_file: scripts/test.sh

set -e

echo "🧪 Running vidkompy tests..."

# Use python3 explicitly
PYTHON_CMD="python3"

# Install test dependencies
echo "📦 Installing test dependencies..."
$PYTHON_CMD -m pip install --break-system-packages -e ".[test]"

# Run linting
echo "🔍 Running linting..."
$PYTHON_CMD -m ruff check src/vidkompy tests

# Run type checking
echo "🔍 Running type checking..."
$PYTHON_CMD -m mypy src/vidkompy tests

# Run tests with coverage
echo "🧪 Running tests with coverage..."
$PYTHON_CMD -m pytest --cov-report=term-missing --cov-config=pyproject.toml --cov=src/vidkompy --cov=tests tests/ -v

# Run formatting check
echo "🔍 Checking formatting..."
$PYTHON_CMD -m ruff format --check src/vidkompy tests

echo "✅ All tests passed!"