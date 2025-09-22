#!/bin/bash
set -e

# Check if the application is responding
if curl -f -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "Health check passed"
    exit 0
else
    echo "Health check failed"
    exit 1
fi