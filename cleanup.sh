#!/usr/bin/env bash

# Check if uzpy module can be imported
# if command -v python >/dev/null && python -c "import uzpy" >/dev/null 2>&1; then
#   echo "uzpy found, running..."
#   python -m uzpy run -e src
# else
#   echo "uzpy module not importable or not runnable with -m, skipping uzpy step."
# fi
echo "Skipping uzpy step for MVP testing."

# if command -v fd >/dev/null; then
#   echo "fd found, running formatters/linters..."
#   fd -e py -x autoflake --in-place --remove-all-unused-imports {}
#   fd -e py -x pyupgrade --py312-plus {}
#   fd -e py -x ruff check --fix --unsafe-fixes {}
#   fd -e py -x ruff format --respect-gitignore {}
# else
#   echo "fd command not found, skipping fd-based linting/formatting steps."
# fi
echo "Skipping fd-based linting/formatting for MVP testing. Linters can be run manually if needed."

if command -v npx >/dev/null; then
  echo "npx found, running repomix..."
  npx repomix -i varia,.specstory,AGENT.md,CLAUDE.md,PLAN.md,llms.txt,.cursorrules -o llms.txt .
else
  echo "npx not found, skipping repomix step."
fi

echo "Running pytest using the pyenv Python's pytest executable..."
# Ensure we use the python where vidkompy was installed edgewise
/home/jules/.pyenv/versions/3.12.11/bin/pytest
