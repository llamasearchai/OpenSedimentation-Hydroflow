.PHONY: init install dev test lint format build docker docker-run api

init:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip

install:
	. .venv/bin/activate && pip install -e .

dev:
	. .venv/bin/activate && pip install -e .[dev]

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && black --check hydroflow tests && isort --check-only hydroflow tests

format:
	. .venv/bin/activate && black hydroflow tests && isort hydroflow tests

build:
	. .venv/bin/activate && python -m build

docker:
	docker build -f docker/Dockerfile -t hydroflow-api .

docker-run:
	docker run --rm -p 8000:8000 hydroflow-api

api:
	. .venv/bin/activate && hydroflow api serve --host 0.0.0.0 --port 8000
