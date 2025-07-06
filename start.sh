#!/usr/bin/env zsh

MONGO_HOST=localhost uvicorn main:app --host 0.0.0.0 --port 8001 --reload
