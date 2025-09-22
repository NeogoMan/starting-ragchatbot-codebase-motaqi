#!/bin/bash
# Format all Python code using black and isort

echo "🎨 Formatting code with black..."
uv run black .

echo "📚 Sorting imports with isort..."
uv run isort .

echo "✅ Code formatting complete!"