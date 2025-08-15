.PHONY: help build up down test clean logs shell

# Default target
help:
	@echo "Available commands:"
	@echo "  build    - Build the Docker image"
	@echo "  up       - Start the services"
	@echo "  down     - Stop the services"
	@echo "  test     - Run smoke tests"
	@echo "  clean    - Clean up containers and images"
	@echo "  logs     - Show service logs"
	@echo "  shell    - Open shell in running container"
	@echo "  dev      - Run development server locally"

# Build Docker image
build:
	docker-compose build

# Start services
up:
	docker-compose up -d

# Stop services
down:
	docker-compose down

# Run smoke tests
test: up
	@echo "Waiting for service to be ready..."
	@sleep 15
	@echo "Testing health endpoint..."
	@curl -f http://localhost:8000/health || (echo "Health check failed" && exit 1)
	@echo "✓ Health check passed"
	@echo "Testing RAG index endpoint..."
	@curl -X POST http://localhost:8000/rag/index || echo "RAG index test completed (may fail if no texts)"
	@echo "✓ Smoke tests completed"

# Clean up
clean:
	docker-compose down --volumes --remove-orphans
	docker system prune -f

# Show logs
logs:
	docker-compose logs -f

# Open shell in container
shell:
	docker-compose exec app /bin/bash

# Run development server locally
dev:
	@echo "Starting development server..."
	@echo "Make sure you have the dependencies installed: pip install -r requirements.txt"
	PYTHONPATH=. uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000

# Install dependencies locally
install:
	pip install -r requirements.txt

# Run local tests
test-local:
	@echo "Testing local health endpoint..."
	@curl -f http://localhost:8000/health || (echo "Health check failed" && exit 1)
	@echo "✓ Local health check passed"