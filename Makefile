.PHONY: install migrate run test lint format clean docker-build

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

migrate:
	alembic upgrade head

run:
	uvicorn api.main:app --reload --port ${API_PORT}

test:
	pytest -v tests/

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy .

format:
	black .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

docker-build:
	docker build -t ecotrack:latest .
