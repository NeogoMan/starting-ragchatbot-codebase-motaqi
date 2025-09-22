#!/bin/bash
# Run all quality checks in sequence

echo "🚀 Running complete code quality checks..."
echo ""

echo "1️⃣ Formatting code..."
./scripts/format.sh
echo ""

echo "2️⃣ Running linting..."
./scripts/lint.sh
echo ""

echo "3️⃣ Running tests..."
uv run pytest
echo ""

echo "🎉 All quality checks complete!"