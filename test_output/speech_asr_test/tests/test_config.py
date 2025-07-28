from pathlib import Path

import pytest

from speech_asr_test.config import Settings, get_settings


def test_settings_from_yaml():
    settings = get_settings()
    
    assert settings.model.checkpoint == "openai/whisper-small"
    assert settings.data.name == "mozilla-foundation/common_voice_11_0"
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
    