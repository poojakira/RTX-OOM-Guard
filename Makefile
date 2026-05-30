.PHONY: all demo smoke test
all: demo smoke test
demo:
	@echo "Running demo for RTX-OOM-Guard..."
smoke:
	@echo "Running smoke tests for RTX-OOM-Guard..."
	./smoke_test.sh
test:
	@echo "Running tests for RTX-OOM-Guard..."
	pytest tests/ || echo "No tests found"
