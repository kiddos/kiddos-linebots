kiddos-linebots
===============

```sh
cp .env.example .env
docker build -t kiddos-linebots .
docker run -d -p 8001:8001 --name kiddos-linebots -v $(pwd)/data:/app/data --add-host host.docker.internal:host-gateway kiddos-linebots
```

```
./start.sh
```
