#!/usr/bin/env zsh

MONGO_HOST=localhost CHROMA_HOST=localhost CHROMA_PORT=8001 uvicorn main:app --host 0.0.0.0 --port 8003 --reload
