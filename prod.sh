#!/usr/bin/env sh

uvicorn main:app --host 0.0.0.0 --port 8003 --log-config log.ini
