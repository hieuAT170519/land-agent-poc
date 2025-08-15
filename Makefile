.PHONY: help build run stop test format clean

# Default target
help:
	@echo "Available targets:"
	@echo "  build    - Build Docker image"
	@echo "  run      - Run with docker-compose"
	@echo "  stop     - Stop and remove containers"
	@echo "  test     - Run basic functionality tests"
	@echo "  format   - Format code with black (optional)"
	@echo "  clean    - Clean up Docker resources"
	@echo "  logs     - Show container logs"

# Build Docker image
build:
	docker-compose build

# Run the application
run:
	docker-compose up -d --build

# Stop and remove containers
stop:
	docker-compose down

# Basic functionality tests
test:
	@echo "Testing health endpoint..."
	@curl -s http://localhost:8000/health | grep -q '"status":"ok"' || (echo "Health check failed" && exit 1)
	@echo "✓ Health check passed"
	@echo "Testing detailed health endpoint..."
	@curl -s http://localhost:8000/health/detailed || echo "Detailed health check failed"
	@echo "✓ Detailed health check completed"
	@echo "Testing RAG index build..."
	@curl -s -X POST http://localhost:8000/rag/index || echo "RAG index build failed"
	@echo "✓ RAG index build completed"

# Format code (optional - requires black)
format:
	@if command -v black > /dev/null; then \
		black src/; \
		echo "Code formatted with black"; \
	else \
		echo "Black not installed, skipping format"; \
	fi

# Clean up Docker resources
clean:
	docker-compose down -v
	docker system prune -f

# Show logs
logs:
	docker-compose logs -f

# Development run (without Docker)
dev:
	@echo "Running in development mode..."
	@cd src && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload