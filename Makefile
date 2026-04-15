.PHONY: build up down logs seed test shell models

COMPOSE ?= docker compose

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

seed:
	python scripts/seed_db.py

test:
	pytest tests/

shell:
	$(COMPOSE) exec admin python

models:
	python scripts/download_models.py --skip-insightface