"""
ML-centric configuration management with modality-aware settings.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# Modality-specific model configurations
class NLPModelConfig(BaseModel):
    """Configuration for NLP models."""
    checkpoint: str = "{{ cookiecutter.model_checkpoint.nlp if cookiecutter.modality == 'nlp' else 'distilbert-base-uncased' }}"
    max_length: int = 512
    num_labels: int = 2
    dropout: float = 0.1

class SpeechASRModelConfig(BaseModel):
    """Configuration for Automatic Speech Recognition (Whisper)."""
    checkpoint: str = "{{ cookiecutter.model_checkpoint.speech_asr if cookiecutter.modality == 'speech' and cookiecutter.speech_task == 'asr' else 'openai/whisper-small' }}"
    sample_rate: int = 16000
    max_audio_length: int = 30  # seconds
    language: str = "en"
    task: Literal["transcribe", "translate"] = "transcribe"

class SpeechTTSModelConfig(BaseModel):
    """Configuration for Text-to-Speech (CSM)."""
    checkpoint: str = "{{ cookiecutter.model_checkpoint.speech_tts if cookiecutter.modality == 'speech' and cookiecutter.speech_task == 'tts' else 'sesame/csm-1b' }}"
    sample_rate: int = 24000  # CSM uses 24kHz
    max_tokens: int = 1024
    temperature: float = 0.8
    voice_preset: str = "default"

class VisionModelConfig(BaseModel):
    """Configuration for vision models."""
    checkpoint: str = "{{ cookiecutter.model_checkpoint.vision if cookiecutter.modality == 'vision' else 'google/vit-base-patch16-224' }}"
    image_size: int = 224
    num_labels: int = 10  # CIFAR-10 default
    patch_size: int = 16
    dropout: float = 0.1


class ExperimentConfig(BaseModel):
    """Core experiment configuration - ML researcher focused."""
    name: str = "{{ cookiecutter.project_slug }}_experiment"
    seed: int = 42
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
class DataConfig(BaseModel):
    """Data configuration."""
    {% if cookiecutter.modality == 'speech' -%}
    name: str = "{{ cookiecutter.dataset_name['speech_' + cookiecutter.speech_task] if 'speech_' + cookiecutter.speech_task in cookiecutter.dataset_name else 'mozilla-foundation/common_voice_11_0' }}"
    {% else -%}
    name: str = "{{ cookiecutter.dataset_name[cookiecutter.modality] if cookiecutter.modality in cookiecutter.dataset_name else 'imdb' }}"
    {% endif -%}
    train_split: str = "train"
    eval_split: str = "test"
    validation_split: Optional[str] = None
    max_samples: Optional[int] = None  # For quick iterations
    preprocessing_num_workers: int = 4

class TrainingConfig(BaseModel):
    """Training hyperparameters - ML researcher focused."""
    # Core training params
    epochs: int = Field(3, description="Number of training epochs")
    batch_size: int = Field(16, description="Per-device batch size")
    eval_batch_size: int = Field(32, description="Per-device eval batch size")
    
    # Learning rate and optimization
    learning_rate: float = Field(3e-4, description="Peak learning rate")
    weight_decay: float = Field(0.01, description="Weight decay for regularization")
    warmup_ratio: float = Field(0.1, description="Warmup ratio of total steps")
    lr_scheduler: Literal["linear", "cosine", "constant"] = "linear"
    
    # Evaluation and saving
    eval_strategy: Literal["steps", "epoch", "no"] = "epoch"
    eval_steps: Optional[int] = None
    save_strategy: Literal["steps", "epoch", "no"] = "epoch"
    save_steps: Optional[int] = None
    save_total_limit: int = 2
    
    # Logging and monitoring
    logging_steps: int = Field(50, description="Log every N steps")
    log_predictions: bool = Field(True, description="Log sample predictions during eval")
    
    # Early stopping
    early_stopping: bool = Field(False, description="Enable early stopping")
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Advanced training
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 0


class InferenceConfig(BaseModel):
    """Inference and serving configuration."""
    # Model serving
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    batch_timeout: float = 0.1
    max_batch_size: int = 8
    
    # Inference parameters
    temperature: float = Field(1.0, description="Sampling temperature")
    top_k: Optional[int] = Field(None, description="Top-k sampling")
    top_p: Optional[float] = Field(None, description="Top-p (nucleus) sampling")
    do_sample: bool = Field(False, description="Use sampling instead of greedy")
    
    # Device optimization
    torch_compile: bool = Field(False, description="Use torch.compile for speedup")
    quantization: Optional[Literal["int8", "int4"]] = Field(None, description="Model quantization")


class ComputeConfig(BaseModel):
    """Compute and resource configuration."""
    # Local compute
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    fp16: bool = Field(True, description="Use FP16 mixed precision")
    tf32: bool = Field(True, description="Use TF32 on Ampere GPUs")
    
    # Memory optimization
    gradient_checkpointing: bool = Field(True, description="Save memory with gradient checkpointing")
    dataloader_pin_memory: bool = Field(True, description="Pin memory for faster data loading")
    
    {% if cookiecutter.cloud_provider != 'none' -%}
    # Cloud compute (optional)
    cloud_provider: Literal["gcp", "aws", "azure"] = "{{ cookiecutter.cloud_provider }}"
    instance_type: str = "g5.xlarge"  # Default GPU instance
    region: str = "us-west-2"
    spot_instances: bool = Field(True, description="Use spot instances for cost savings")
    {% endif -%}


class Settings(BaseSettings):
    """Main settings class - modality-aware and ML-researcher friendly."""
    # Core experiment setup
    modality: Literal["nlp", "speech", "vision"] = "{{ cookiecutter.modality }}"
    experiment: ExperimentConfig = ExperimentConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()
    compute: ComputeConfig = ComputeConfig()
    
    # Modality-specific model config (set dynamically)
    model: Any = None
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set modality-specific model config
        {% if cookiecutter.modality == 'nlp' -%}
        self.model = NLPModelConfig()
        {% elif cookiecutter.modality == 'speech' -%}
        {% if cookiecutter.speech_task == 'asr' -%}
        self.model = SpeechASRModelConfig()
        {% elif cookiecutter.speech_task == 'tts' -%}
        self.model = SpeechTTSModelConfig()
        {% endif -%}
        {% elif cookiecutter.modality == 'vision' -%}
        self.model = VisionModelConfig()
        {% endif -%}
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "Settings":
        """Load config from YAML with modality-aware model selection."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        settings = cls(**config_dict)
        
        # Override model config if specified in YAML
        if "model" in config_dict:
            modality = config_dict.get("modality", "{{ cookiecutter.modality }}")
            if modality == "nlp":
                settings.model = NLPModelConfig(**config_dict["model"])
            elif modality == "speech":
                {% if cookiecutter.speech_task == 'asr' -%}
                settings.model = SpeechASRModelConfig(**config_dict["model"])
                {% elif cookiecutter.speech_task == 'tts' -%}
                settings.model = SpeechTTSModelConfig(**config_dict["model"])
                {% endif -%}
            elif modality == "vision":
                settings.model = VisionModelConfig(**config_dict["model"])
        
        return settings
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary for logging/tracking."""
        return {
            "modality": self.modality,
            "experiment": self.experiment.dict(),
            "data": self.data.dict(),
            "training": self.training.dict(),
            "model": self.model.dict() if hasattr(self.model, 'dict') else str(self.model),
            "compute": self.compute.dict()
        }


def get_settings() -> Settings:
    """Get settings with automatic config file detection."""
    config_path = Path(__file__).parent.parent / "configs" / "settings.yaml"
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return Settings()
    return Settings.from_yaml(config_path)


def create_experiment_config(
    name: str,
    learning_rate: float = 3e-4,
    epochs: int = 3,
    batch_size: int = 16,
    **kwargs
) -> Settings:
    """Quick experiment config creation - researcher-friendly helper."""
    settings = get_settings()
    settings.experiment.name = name
    settings.training.learning_rate = learning_rate
    settings.training.epochs = epochs
    settings.training.batch_size = batch_size
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(settings.training, key):
            setattr(settings.training, key, value)
        elif hasattr(settings.model, key):
            setattr(settings.model, key, value)
    
    return settings