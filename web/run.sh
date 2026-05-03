#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
echo "=== AVM Web UI ==="
echo "http://127.0.0.1:8080"
echo
exec python -m web.server
