start:
	 uvicorn api:app --host 0.0.0.0 --port 8000

up:
	docker compose up -d

build:
	docker compose build

down:
	docker compose down

install:
	docker run --rm --interactive --tty -v .:/app composer:2.4 install --ignore-platform-reqs

push:
	docker buildx build --platform linux/amd64 -t edk24/face-api:1.0 . --push