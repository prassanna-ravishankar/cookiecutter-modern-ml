from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    checkpoint: str = "{{ cookiecutter.model_checkpoint }}"
    max_length: int = 512


class TrainingConfig(BaseModel):
    dataset_name: str = "imdb"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_steps: int = 10
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


class CloudConfig(BaseModel):
    provider: Literal["gcp", "aws", "azure"] = "{{ cookiecutter.default_cloud }}"
    instance_type: str = "g5.xlarge"
    region: str = "us-west-2"


class Settings(BaseSettings):
    model: ModelConfig
    training: TrainingConfig
    serving: ServingConfig
    cloud: CloudConfig

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Settings":
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


def get_settings() -> Settings:
    config_path = Path(__file__).parent.parent / "configs" / "settings.yaml"
    return Settings.from_yaml(config_path)