# Modern ML Cookiecutter

🚀 A modality-aware, end-to-end template for modern machine learning projects covering **NLP**, **Speech**, and **Vision** with best-in-class models and researcher-friendly configuration.

## ✨ Features

- **🎯 Multi-Modal Support**: Choose from NLP (DistilBERT), Speech (Whisper), or Vision (ViT) with optimized configurations
- **⚡ Fast Dependency Management**: Uses [uv](https://github.com/astral-sh/uv) for lightning-fast package management
- **🧠 ML-Centric Configuration**: Researcher-friendly parameter names (`epochs` not `num_train_epochs`)
- **🖥️ Mac MPS Support**: Optimized for Apple Silicon with Metal Performance Shaders
- **☁️ Local & Cloud Training**: Seamless training with Hugging Face Accelerate locally or SkyPilot in the cloud
- **🚀 Production-Ready Serving**: High-performance model serving with LitServe
- **📊 Experiment Tracking**: Optional integration with [tracelet](https://github.com/prassanna-ravishankar/tracelet)
- **🔧 Type-Safe Configuration**: Pydantic-based settings with modality-aware validation

## 🎯 Supported Modalities

| Modality | Task | Model | Dataset | Key Libraries |
|----------|------|-------|---------|---------------|
| **NLP** | Text Classification | DistilBERT | IMDB | transformers, datasets |
| **Speech** | ASR (Speech-to-Text) | Whisper | Common Voice | transformers, librosa, torchaudio |
| **Speech** | TTS (Text-to-Speech) | CSM (Sesame) | Conversational | transformers, CSM |
| **Vision** | Image Classification | Vision Transformer | CIFAR-10 | torchvision, PIL, opencv |

## 🚀 Quick Start

1. Install cookiecutter:
```bash
uv tool install cookiecutter
# or: pip install cookiecutter
```

2. Generate a new project:
```bash
cookiecutter https://github.com/prassanna-ravishankar/cookiecutter-modern-ml
```

3. Choose your modality and configuration:
```
[1/12] project_name (My ML Project): Voice Assistant
[2/12] Select modality:
  1 - nlp
  2 - speech  
  3 - vision
  Choose from [1/2/3] (1): 2
[3/12] Select speech_task:
  1 - asr
  2 - tts
  Choose from [1/2] (1): 2
[4/12] Select use_tracelet:
  1 - yes
  2 - no
```

4. Start developing:
```bash
cd voice_assistant
uv sync
uv run task train
```

## 📋 What's Included

### Project Structure
```
your_project/
├── .github/workflows/    # Modern CI with astral-sh/setup-uv
├── configs/             # ML-centric YAML configuration
├── models/              # Trained model artifacts  
├── notebooks/           # Jupyter notebooks (optional)
├── your_package/        # Source code
│   ├── config.py        # Modality-aware configuration
│   ├── data_utils.py    # Polars-based data processing
│   ├── deployment/      # LitServe model serving
│   └── models/          # Modality-specific training
├── tests/               # Pytest test suite
├── pyproject.toml       # Modern Python packaging
└── sky_task.yaml        # Cloud training config
```

### Pre-configured Tools

- **uv**: Ultra-fast Python package management
- **Transformers**: State-of-the-art models for all modalities
- **Accelerate**: Multi-device training (CUDA/MPS/CPU)
- **LitServe**: High-performance model serving
- **Polars**: Fast data processing (not pandas)
- **Pydantic**: Type-safe configuration management
- **Ruff**: Fast Python linter and formatter
- **Pytest**: Testing framework

## 🎯 Example Workflows

### NLP: Sentiment Analysis
```bash
# Train DistilBERT on IMDB
uv run task train

# Serve the model
uv run task serve

# Test inference
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'
```

### Vision: Image Classification  
```bash
# Train ViT on CIFAR-10
uv run task train

# The model automatically detects Mac MPS, CUDA, or CPU
# Batch sizes adjust automatically for memory constraints
```

### Speech: ASR (Whisper) or TTS (CSM)
```bash
# ASR: Train Whisper for speech-to-text
uv run task train

# TTS: Train CSM for conversational speech generation  
uv run task train

# Automatically handles audio preprocessing and evaluation metrics
```

## 🔧 Configuration

### ML-Researcher Friendly Settings

```yaml
# configs/settings.yaml
modality: "nlp"

experiment:
  name: "bert_baseline"
  seed: 42

training:
  epochs: 5           # Not num_train_epochs!
  batch_size: 32      # Not per_device_train_batch_size!
  learning_rate: 3e-4
  warmup_ratio: 0.1

model:
  checkpoint: "distilbert-base-uncased"
  max_length: 512
  dropout: 0.1

compute:
  device: "auto"      # Automatically detects MPS/CUDA/CPU
  fp16: true
  gradient_checkpointing: true
```

### Quick Experiment Setup

```python
from your_package.config import create_experiment_config

# Researcher-friendly experiment creation
config = create_experiment_config(
    name="distilbert_large_lr",
    learning_rate=5e-4,
    epochs=10,
    batch_size=64
)
```

### Modality-Specific Features

- **NLP**: Automatic tokenizer padding, sequence classification metrics
- **Speech ASR**: 16kHz audio processing, WER metrics, Whisper optimizations  
- **Speech TTS**: 24kHz generation, naturalness metrics, CSM conversational features
- **Vision**: Image preprocessing, patch-based transformers, classification metrics

## 📊 Device Optimization

The template automatically optimizes for your hardware:

- **Mac MPS**: Optimized batch sizes, no fp16, proper memory pinning
- **CUDA**: Full fp16, TensorFloat-32, optimal batch sizes
- **CPU**: Conservative batch sizes, fp32, minimal workers

## ☁️ Cloud Training

Deploy to any cloud with SkyPilot:

```bash
uv run task train-cloud
```

Supports AWS, GCP, Azure with spot instance optimization.

## 🧪 Testing

Run the full test suite:

```bash
python3 self_test.py  # Validate template completeness
uv run task test      # Run generated project tests
uv run task lint      # Code quality checks
```

## 📚 Documentation

- [uv Documentation](https://github.com/astral-sh/uv)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)  
- [LitServe Documentation](https://github.com/Lightning-AI/litserve)
- [SkyPilot Documentation](https://docs.skypilot.co/)
- [Tracelet Experiment Tracking](https://github.com/prassanna-ravishankar/tracelet)

## 🎨 Design Philosophy

- **Simplicity over Features**: Avoid over-engineering, focus on researcher needs
- **ML-Centric**: Parameter names and structure match ML research conventions
- **Modality-Aware**: Each domain (NLP/Speech/Vision) has optimized defaults
- **Modern Tooling**: Latest best practices (uv, Polars, LitServe, Pydantic)
- **Mac-First**: Optimized for Apple Silicon development

## 🤝 Contributing

Contributions welcome! This template prioritizes simplicity and researcher experience.

## 📄 License

MIT License - build amazing ML projects!