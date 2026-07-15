.PHONY: install install-dev train run test lint clean format

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

train:
	python scripts/train_toy_model.py

run:
	uvicorn app.main:app --reload

test:
	pytest tests/ -v --tb=short

lint:
	flake8 . --max-line-length=120
	mypy . --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage

format:
	black .
	isort .
