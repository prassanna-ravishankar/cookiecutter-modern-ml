# {{ cookiecutter.project_name }}

A modern machine learning project built with the Modern ML Cookiecutter template.

## 🚀 Quick Start

### Prerequisites

- Python {{ cookiecutter.python_version }}+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install project dependencies:
```bash
uv sync --all-extras
```

## 📊 Training

### Local Training

Train the model locally using Accelerate:

```bash
uv run task train
```

{% if cookiecutter.modality == 'nlp' -%}This will fine-tune a `{{ cookiecutter.model_checkpoint.nlp }}` model on the {{ cookiecutter.dataset_name.nlp }} dataset for sentiment classification.{% elif cookiecutter.modality == 'speech' and cookiecutter.speech_task == 'asr' -%}This will fine-tune a `{{ cookiecutter.model_checkpoint.speech_asr }}` model on the {{ cookiecutter.dataset_name.speech_asr }} dataset for automatic speech recognition.{% elif cookiecutter.modality == 'speech' and cookiecutter.speech_task == 'tts' -%}This will fine-tune a `{{ cookiecutter.model_checkpoint.speech_tts }}` model on the {{ cookiecutter.dataset_name.speech_tts }} dataset for text-to-speech generation.{% elif cookiecutter.modality == 'vision' -%}This will fine-tune a `{{ cookiecutter.model_checkpoint.vision }}` model on the {{ cookiecutter.dataset_name.vision }} dataset for image classification.{% endif %}

{% if cookiecutter.cloud_provider != "none" -%}
### Cloud Training

Deploy training to the cloud using SkyPilot:

```bash
uv run task train-cloud
```
{% endif %}

This will launch a training job on {{ cookiecutter.cloud_provider }} using the configuration in `sky_task.yaml`.

## 🔧 Development

### Code Quality

Run linting:
```bash
uv run task lint
```

Format code:
```bash
uv run task format
```

Type checking:
```bash
uv run typecheck
```

### Testing

Run tests:
```bash
uv run task test
```

## 🚢 Deployment

After training, serve the model as a REST API:

```bash
uv run task serve
```

The API will be available at `http://localhost:8000`.

### API Usage

Send a POST request to classify text:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

Response:
```json
{
  "prediction": "positive",
  "confidence": 0.98,
  "probabilities": {
    "negative": 0.02,
    "positive": 0.98
  }
}
```

## 📁 Project Structure

```
.
├── configs/           # Configuration files
├── models/           # Trained model artifacts
├── notebooks/        # Jupyter notebooks
├── src/              # Source code
│   ├── config.py     # Configuration management
│   ├── deployment/   # Model serving code
│   └── models/       # Training scripts
├── tests/            # Test files
└── sky_task.yaml     # SkyPilot configuration
```

## 🛠️ Configuration

All project settings are managed in `configs/settings.yaml`. The configuration is loaded using Pydantic for type safety and validation.

## 📝 License

This project was created using the Modern ML Cookiecutter template.