# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
- Install dependencies: `uv sync --all-extras`
- Update dependencies: `uv add <package>` or `uv add --dev <package>`

### Code Quality
- Lint code: `uv run lint` (runs ruff check)
- Format code: `uv run format` (runs ruff format)
- Type check: `uv run typecheck` (runs mypy)
- Run tests: `uv run test` (runs pytest)

### Training & Deployment
- Train locally: `uv run train-local` (uses Accelerate)
- Train on cloud: `uv run train-cloud` (uses SkyPilot)
- Serve model: `uv run serve` (starts LitServe API)

## Architecture

This is a cookiecutter template for modern ML projects with the following structure:

### Template Variables (cookiecutter.json)
- `project_name`: Human-readable project name
- `project_slug`: Python-safe project identifier
- `author_name`, `author_email`: Project author details
- `model_checkpoint`: Hugging Face model to fine-tune
- `default_cloud`: Cloud provider (gcp/aws/azure)
- `python_version`: Python version requirement

### Core Components

1. **Dependency Management**: Uses `uv` with `pyproject.toml` for fast, reproducible environments

2. **Configuration**: 
   - `configs/settings.yaml`: Central configuration file
   - `src/config.py`: Pydantic models for type-safe config access

3. **Training Pipeline**:
   - `src/models/train_model.py`: Hugging Face Accelerate-based training
   - Supports local (CPU/GPU/MPS) and cloud training
   - Example: Fine-tuning DistilBERT on IMDB dataset

4. **Deployment**:
   - `src/deployment/serve.py`: LitServe-based model serving
   - High-performance REST API for inference

5. **Cloud Integration**:
   - `sky_task.yaml`: SkyPilot configuration for cloud training
   - Supports AWS, GCP, and Azure

6. **CI/CD**:
   - `.github/workflows/ci.yaml`: GitHub Actions for testing and linting
   - Automated quality checks on push/PR

### Key Design Decisions

- **uv over pip/poetry**: Fastest Python package manager
- **Accelerate over custom**: Seamless multi-device training
- **LitServe over FastAPI**: Optimized for ML model serving
- **SkyPilot over custom scripts**: Declarative cloud orchestration
- **Pydantic settings**: Type-safe configuration management