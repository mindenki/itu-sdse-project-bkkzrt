#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = itu-sdse-project-bkkzrt
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format



## Run tests
.PHONY: test
test:
	uv run pytest


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"


## Install the Dagger CLI into .venv (version pinned by dagger-io in uv.lock)
.PHONY: tools
tools: requirements
	@uv run python -c "$${INSTALL_DAGGER_PYSCRIPT}"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Pull DVC-tracked metadata and download the images
.PHONY: data
data: requirements
	uv run python -m vial_classifier.dataset

## Train the model (logs to the local MLflow store in mlflow-state/)
.PHONY: train
train: data
	uv run python -m vial_classifier.modeling.train

## Build and run training in the Linux container (set TRAIN_ARGS for smoke-test options)
.PHONY: train-docker
train-docker:
	docker build -f docker/Dockerfile.train -t vial-train .
	docker run --rm -v "$(CURDIR)/artifacts:/app/artifacts" -v "$(CURDIR)/mlflow-state:/app/mlflow-state" -v "$(CURDIR)/data:/app/data" vial-train $(TRAIN_ARGS)

## Promote the best model and export it to models/model
.PHONY: export
export:
	uv run python -m vial_classifier.modeling.registry


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Uses the dagger-io SDK's own downloader: it fetches the CLI release matching the
# SDK version and verifies its checksum. The download cache lives inside .venv and
# the binary is copied next to python (.venv/bin, or .venv/Scripts on Windows).
define INSTALL_DAGGER_PYSCRIPT
import os, shutil, sys, sysconfig; \
from pathlib import Path; \
os.environ['XDG_CACHE_HOME'] = str(Path(sys.prefix) / 'cache'); \
from dagger.provisioning._download import Downloader; \
src = Path(Downloader().get_sync()); \
dst = Path(sysconfig.get_path('scripts')) / ('dagger.exe' if src.suffix == '.exe' else 'dagger'); \
shutil.copy2(src, dst); \
print(f'Installed {dst}')
endef
export INSTALL_DAGGER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
