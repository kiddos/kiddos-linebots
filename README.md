kiddos-linebots
===============

```sh
cp .env.example .env
./start_docker.sh
```

or manually:

```sh
docker build -t kiddos-linebots:latest .
docker run -d --name kiddos-linebots -v $(pwd)/data:/app/data --add-host=host.docker.internal:host-gateway -p 8003:8003 kiddos-linebots
```

```
./start.sh
```
