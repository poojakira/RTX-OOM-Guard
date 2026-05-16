# Makefile for RTX-OOM-Guard: Predictive GPU Memory Defragmenter

.PHONY: help install test run run-benchmarks dashboard docker-build lint clean

# Default target
help:
	@echo "RTX-OOM-Guard Infrastructure Makefile"
	@echo "-------------------------------------"
	@echo "install         : Install dependencies and the rtx_oom_guard package in editable mode"
	@echo "test            : Run the pytest suite"
	@echo "run             : Start the ML infra monitor (requires configs/config.yaml)"
	@echo "run-benchmarks  : Run the full memory fragmentation benchmark suite"
	@echo "dashboard       : Start the monitoring dashboard"
	@echo "docker-build    : Build the production-grade Docker image"
	@echo "lint            : Run ruff for style and quality checks"
	@echo "clean           : Remove temporary files and build artifacts"

install:
	pip install -e "."

test:
	pytest tests/

run:
	python run.py --config configs/config.yaml

run-benchmarks:
	python run_benchmark.py --config configs/config.yaml

dashboard:
	cd dashboard && npm run dev

docker-build:
	docker build -t rtx-oom-guard:latest .

lint:
	ruff check .

clean:
ifeq ($(OS),Windows_NT)
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d"
	for /d /r . %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d"
	if exist build rd /s /q build
	if exist dist rd /s /q dist
else
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/
endif
	@echo "Cleaned environment."
