from pathlib import Path

import pytest

from speech_tts_test.config import Settings, get_settings


def test_settings_from_yaml():
    settings = get_settings()
    
    assert settings.model.checkpoint == "sesame/csm-1b"
    assert settings.data.name == "conversational_dataset"
    assert settings.training.epochs == 3
    assert settings.inference.port == 8000
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
    