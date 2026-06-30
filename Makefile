.PHONY: install dev test lint serve docker-build docker-run

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/

serve:
	uvicorn google_commands.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker compose build

docker-run:
	docker compose up
