# Modern ML Cookiecutter

🚀 A high-performance, end-to-end template for modern machine learning projects that accelerates development from local prototyping to scalable cloud deployment.

## ✨ Features

- **Fast Dependency Management**: Uses [uv](https://github.com/astral-sh/uv) for lightning-fast package management
- **Local & Cloud Training**: Seamless training with Hugging Face Accelerate locally or SkyPilot in the cloud
- **Production-Ready Serving**: High-performance model serving with LitServe
- **Type-Safe Configuration**: Pydantic-based settings management
- **Code Quality**: Pre-configured with ruff, mypy, and pytest
- **CI/CD Ready**: GitHub Actions workflow included

## 🚀 Quick Start

1. Install cookiecutter:
```bash
pip install cookiecutter
```

2. Generate a new project:
```bash
cookiecutter https://github.com/yourusername/cookiecutter-modern-ml
```

3. Answer the prompts:
- `project_name`: Your project's human-readable name
- `author_name`: Your name
- `model_checkpoint`: Hugging Face model to fine-tune (default: distilbert-base-uncased)
- `default_cloud`: Your preferred cloud provider (gcp/aws/azure)

4. Navigate to your new project and start developing:
```bash
cd your_project_name
uv sync --all-extras
uv run train-local
```

## 📋 What's Included

### Project Structure
```
your_project/
├── .github/workflows/    # CI/CD pipelines
├── configs/             # Configuration files
├── models/              # Trained model artifacts
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code
│   ├── config.py        # Configuration management
│   ├── deployment/      # Model serving
│   └── models/          # Training scripts
├── tests/               # Test files
├── pyproject.toml       # Project dependencies
├── sky_task.yaml        # Cloud training config
└── README.md            # Project documentation
```

### Pre-configured Tools

- **uv**: Ultra-fast Python package management
- **Hugging Face Transformers**: State-of-the-art models
- **Accelerate**: Multi-device training support
- **LitServe**: High-performance model serving
- **SkyPilot**: Cloud-agnostic training orchestration
- **Ruff**: Fast Python linter and formatter
- **Mypy**: Static type checking
- **Pytest**: Testing framework

## 🎯 Example Use Case

The template comes with a complete example that fine-tunes a DistilBERT model on the IMDB dataset for sentiment classification:

1. **Train locally**: `uv run train-local`
2. **Train on cloud**: `uv run train-cloud`
3. **Serve the model**: `uv run serve`
4. **Make predictions**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'
```

## 🔧 Customization

### Change the Default Model

Edit `configs/settings.yaml`:
```yaml
model:
  checkpoint: "bert-base-uncased"  # or any Hugging Face model
```

### Add New Dependencies

```bash
uv add transformers datasets torch
uv add --dev pytest ruff mypy
```

### Configure Cloud Training

Edit `sky_task.yaml` to customize instance types, regions, and resources.

## 📚 Documentation

- [uv Documentation](https://github.com/astral-sh/uv)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [LitServe Documentation](https://github.com/Lightning-AI/litserve)
- [SkyPilot Documentation](https://skypilot.readthedocs.io/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.