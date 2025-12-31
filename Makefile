.PHONY: install dev test clean docker-build docker-run docker-stop docker-clean help

PYTHON := python3
PIP := pip3
DOCKER_IMAGE := brinicle
DOCKER_CONTAINER := brinicle_container
HOST := 0.0.0.0
PORT := 1984

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make dev          - Run development server"
	@echo "  make clean        - Clean up cache and temp files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make docker-run-limitless   - Run Docker container w/o limit"
	@echo "  make docker-stop  - Stop Docker container"
	@echo "  make docker-clean - Remove Docker container and image"
	@echo "  make docker-compose-build - Run docker-compose"
	@echo "  make format       - Format code content"
	@echo "  make upload-pypi  - Update pypi package"
	@echo "  make pybuild      - Python build"

install:
	$(PIP) install -r requirements.txt
	bash build.sh

dev:
	uvicorn ref.api:app --host $(HOST) --port $(PORT) --reload


clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -r wheelhouse

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	docker rm -f brinicle 2>/dev/null || true
	docker run -d \
	  --name $(DOCKER_CONTAINER) \
	  -p $(PORT):1984 \
	  -v ./brinicle_data:/app/data \
	  -e PYTHONUNBUFFERED=1 \
	  -e LOG_LEVEL=INFO \
	  --memory="1g" \
	  --cpus="2.0"  \
	  --restart unless-stopped \
	  $(DOCKER_IMAGE)
	@echo "Container started. API available at http://localhost:$(PORT)"

docker-run-limitless:
	docker rm -f brinicle 2>/dev/null || true
	docker run -d \
	  --name $(DOCKER_CONTAINER) \
	  -p $(PORT):1984 \
	  -v ./brinicle_data:/app/data \
	  -e PYTHONUNBUFFERED=1 \
	  -e LOG_LEVEL=INFO \
	  --restart unless-stopped \
	  $(DOCKER_IMAGE)
	@echo "Container started. API available at http://localhost:$(PORT)"

docker-stop:
	docker stop $(DOCKER_CONTAINER) || true
	docker rm $(DOCKER_CONTAINER) || true

docker-clean: docker-stop
	docker rmi $(DOCKER_IMAGE) || true

docker-logs:
	docker logs -f $(DOCKER_CONTAINER)

docker-shell:
	docker exec -it $(DOCKER_CONTAINER) /bin/bash

docker-compose-build:
	docker-compose up --build

format:
	autoflake brinicle --remove-all-unused-imports --quiet --in-place -r --exclude third_party
	isort brinicle --force-single-line-imports
	black brinicle
	autoflake tests --remove-all-unused-imports --quiet --in-place -r --exclude third_party
	isort tests --force-single-line-imports
	black tests

pybuild:
	cibuildwheel --platform linux

upload-pypi:
	python3 -m twine upload wheelhouse/*
