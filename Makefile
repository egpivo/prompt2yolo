SHELL := /bin/bash
VERSION ?= latest
EXECUTABLE := poetry run

.PHONY: clean install-dev build-docker run-docker

clean: clean-pyc clean-build clean-test-coverage

clean-pyc:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

clean-build:
	rm -fr build/ dist/ .eggs/
	find . -name '*.egg-info' -o -name '*.egg' -exec rm -fr {} +

clean-test-coverage:
	rm -f .coverage
	rm -rf .pytest_cache

install-dev:
	@echo -e "\033[1;92mInstalling development dependencies...\033[0m"
	@$(SHELL) envs/conda/build_conda_env.sh -c prompt2yolo

test:
	@echo "Running tests..."
	@$(EXECUTABLE) pytest --cov=prompt2yolo

build-docker:
	docker build -t prompt2yolo:$(VERSION) .

run-docker:
	docker run -p 8501:8501 prompt2yolo:$(VERSION)
