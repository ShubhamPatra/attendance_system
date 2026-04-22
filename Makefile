.PHONY: seed test models run

seed:
	python scripts/seed_db.py

test:
	pytest tests/

models:
	python scripts/download_models.py --skip-insightface

run:
	python run.py