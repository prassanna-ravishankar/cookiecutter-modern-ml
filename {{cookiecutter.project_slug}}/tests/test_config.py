from pathlib import Path

import pytest

from {{ cookiecutter.package_name }}.config import Settings, get_settings


def test_settings_from_yaml():
    settings = get_settings()
    
    {% if cookiecutter.modality == 'nlp' -%}
    assert settings.model.checkpoint == "{{ cookiecutter.model_checkpoint.nlp }}"
    assert settings.data.name == "{{ cookiecutter.dataset_name.nlp }}"
    {% elif cookiecutter.modality == 'speech' and cookiecutter.speech_task == 'asr' -%}
    assert settings.model.checkpoint == "{{ cookiecutter.model_checkpoint.speech_asr }}"
    assert settings.data.name == "{{ cookiecutter.dataset_name.speech_asr }}"
    {% elif cookiecutter.modality == 'speech' and cookiecutter.speech_task == 'tts' -%}
    assert settings.model.checkpoint == "{{ cookiecutter.model_checkpoint.speech_tts }}"
    assert settings.data.name == "{{ cookiecutter.dataset_name.speech_tts }}"
    {% elif cookiecutter.modality == 'vision' -%}
    assert settings.model.checkpoint == "{{ cookiecutter.model_checkpoint.vision }}"
    assert settings.data.name == "{{ cookiecutter.dataset_name.vision }}"
    {% endif -%}
    assert settings.training.epochs == 3
    assert settings.inference.port == 8000
    {% if cookiecutter.cloud_provider != 'none' -%}
    assert settings.compute.cloud_provider == "{{ cookiecutter.cloud_provider }}"
    {% endif -%}


def test_settings_structure():
    settings = get_settings()
    
    # Test that all required attributes exist
    assert hasattr(settings, "model")
    assert hasattr(settings, "training")
    assert hasattr(settings, "inference")
    assert hasattr(settings, "compute")
    assert hasattr(settings, "experiment")
    assert hasattr(settings, "data")
    
    # Test nested attributes
    assert hasattr(settings.model, "checkpoint")
    assert hasattr(settings.training, "epochs")
    assert hasattr(settings.inference, "host")
    {% if cookiecutter.cloud_provider != 'none' -%}
    assert hasattr(settings.compute, "cloud_provider")
    {% endif -%}