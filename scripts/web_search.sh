#!/usr/bin/env bash
# Web search CLI using Ollama's web search API.
#
# Delegates to the Python CLI for consistent formatting, validation,
# and error handling. All flags are passed through to the Python CLI.
#
# Usage:
#   scripts/web_search.sh "your search query"
#   scripts/web_search.sh "your search query" --format markdown
#   scripts/web_search.sh "your search query" --format json --max-results 3
#
# Requires OLLAMA_API_KEY environment variable (or a .env file in the project root).
# Run with --verbose to see structured JSON logging.

set -euo pipefail

QUERY="${1:?Usage: web_search.sh \"search query\" [--format text|json|markdown] [--max-results N] [--verbose]}"
shift

exec uv run python -m sentimentizer.agent.websearch "$QUERY" "$@"