#!/usr/bin/env bash
# this_file: scripts/dev-setup.sh

set -e

echo "🔧 Setting up development environment for vidkompy..."

# Install package in development mode
echo "📦 Installing vidkompy in development mode..."
python3 -m pip install --break-system-packages -e ".[dev,test]"

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
python3 -m pre_commit install

# Run initial formatting and linting
echo "🔧 Running initial formatting and linting..."
python3 -m ruff format src/vidkompy tests
python3 -m ruff check --fix src/vidkompy tests

# Run tests to ensure everything works
echo "🧪 Running tests to verify setup..."
python3 -m pytest tests/ -v --tb=short

echo "✅ Development environment setup complete!"
echo ""
echo "📋 Available commands:"
echo "   ./scripts/test.sh          - Run all tests"
echo "   ./scripts/build.sh         - Build package"
echo "   ./scripts/release.sh X.Y.Z - Create release"
echo "   hatch run test             - Run tests with hatch"
echo "   hatch run fix              - Fix formatting and linting"