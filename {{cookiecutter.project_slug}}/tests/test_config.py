from pathlib import Path

import pytest

from src.config import Settings, get_settings


def test_settings_from_yaml():
    settings = get_settings()
    
    assert settings.model.checkpoint == "{{ cookiecutter.model_checkpoint }}"
    assert settings.model.max_length == 512
    assert settings.training.dataset_name == "imdb"
    assert settings.training.num_train_epochs == 3
    assert settings.serving.port == 8000
    assert settings.cloud.provider == "{{ cookiecutter.default_cloud }}"


def test_settings_structure():
    settings = get_settings()
    
    # Test that all required attributes exist
    assert hasattr(settings, "model")
    assert hasattr(settings, "training")
    assert hasattr(settings, "serving")
    assert hasattr(settings, "cloud")
    
    # Test nested attributes
    assert hasattr(settings.model, "checkpoint")
    assert hasattr(settings.training, "num_train_epochs")
    assert hasattr(settings.serving, "host")
    assert hasattr(settings.cloud, "instance_type")