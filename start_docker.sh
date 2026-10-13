#!/usr/bin/env sh
docker build -t kiddos-linebots:latest .
docker run -d --name kiddos-linebots --add-host=host.docker.internal:host-gateway -p 8003:8003 kiddos-linebots
