#!/usr/bin/env sh

echo "$OLLAMA_HOST"
uvicorn main:app --host 0.0.0.0 --port 8003 --log-config log.ini
