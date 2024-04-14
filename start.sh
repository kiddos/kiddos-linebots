#!/usr/bin/env zsh

# uvicorn main:app --host 0.0.0.0 --port 10080 --reload

# uvicorn main:app --host 0.0.0.0 --port 10442 --reload --ssl-keyfile ~/.local/selfsigned.key --ssl-certfile ~/.local/selfsigned.crt
uvicorn main:app --host 0.0.0.0 --port 10442 --reload --ssl-keyfile privkey.pem --ssl-certfile fullchain.pem 
