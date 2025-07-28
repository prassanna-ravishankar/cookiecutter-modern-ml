#!/usr/bin/env python3
"""
Self-test script to verify the cookiecutter template is complete and self-contained.
"""

import json
from pathlib import Path

def test_cookiecutter_structure():
    """Test that all essential files exist."""
    base_dir = Path(__file__).parent
    template_dir = base_dir / "{{cookiecutter.project_slug}}"
    
    # Check cookiecutter.json exists and is valid
    cookiecutter_json = base_dir / "cookiecutter.json"
    assert cookiecutter_json.exists(), "cookiecutter.json missing"
    
    with open(cookiecutter_json) as f:
        config = json.load(f)
    
    # Verify essential config keys
    required_keys = [
        "project_name", "project_slug", "package_name", 
        "author_name", "model_checkpoint", "dataset_name"
    ]
    for key in required_keys:
        assert key in config, f"Missing required key: {key}"
    
    # Check template directory exists
    assert template_dir.exists(), "Template directory missing"
    
    # Check essential template files
    essential_files = [
        "pyproject.toml",
        "{{cookiecutter.package_name}}/__init__.py",
        "{{cookiecutter.package_name}}/config.py",
        "{{cookiecutter.package_name}}/models/train_model.py",
        "{{cookiecutter.package_name}}/deployment/serve.py",
        "{{cookiecutter.package_name}}/data_utils.py",
        "configs/settings.yaml",
        "tests/test_config.py",
        ".github/workflows/ci.yaml"
    ]
    
    for file_path in essential_files:
        full_path = template_dir / file_path
        assert full_path.exists(), f"Missing essential file: {file_path}"
    
    print("✅ All essential files present")

def test_config_simplicity():
    """Test that config is simple and researcher-friendly."""
    base_dir = Path(__file__).parent
    cookiecutter_json = base_dir / "cookiecutter.json"
    
    with open(cookiecutter_json) as f:
        config = json.load(f)
    
    # Should have minimal config options (< 15)
    assert len(config) <= 15, f"Too many config options: {len(config)}"
    
    # Type checker should default to "none"
    if "type_checker" in config:
        type_checker_options = config["type_checker"]
        if isinstance(type_checker_options, list):
            assert type_checker_options[0] == "none", "Type checker should default to 'none'"
    
    print("✅ Configuration is researcher-friendly")

def test_dependencies():
    """Test that dependencies are reasonable."""
    base_dir = Path(__file__).parent
    pyproject_path = base_dir / "{{cookiecutter.project_slug}}" / "pyproject.toml"
    
    with open(pyproject_path) as f:
        content = f.read()
    
    # Should have core ML dependencies
    required_deps = [
        "transformers", "datasets", "torch", "accelerate", 
        "litserve", "polars", "pydantic"
    ]
    
    for dep in required_deps:
        assert dep in content, f"Missing dependency: {dep}"
    
    # Should conditionally include tracelet
    assert "tracelet" in content, "Missing tracelet integration"
    
    print("✅ Dependencies look good")

def test_tracelet_integration():
    """Test that tracelet is properly integrated."""
    base_dir = Path(__file__).parent
    train_script = base_dir / "{{cookiecutter.project_slug}}" / "{{cookiecutter.package_name}}" / "models" / "train_model.py"
    
    with open(train_script) as f:
        content = f.read()
    
    # Should have tracelet import and usage
    assert "tracelet" in content, "Missing tracelet integration"
    assert "experiment_name" in content, "Missing experiment naming"
    
    print("✅ Tracelet integration present")

def test_code_quality():
    """Test that generated code follows good practices."""
    base_dir = Path(__file__).parent
    template_dir = base_dir / "{{cookiecutter.project_slug}}"
    
    # Check Python files for basic quality
    python_files = [
        "{{cookiecutter.package_name}}/models/train_model.py",
        "{{cookiecutter.package_name}}/deployment/serve.py", 
        "{{cookiecutter.package_name}}/data_utils.py",
        "{{cookiecutter.package_name}}/config.py"
    ]
    
    for file_path in python_files:
        full_path = template_dir / file_path
        with open(full_path) as f:
            content = f.read()
        
        # Basic checks
        assert "import logging" in content, f"Missing logging import in {file_path}"
        assert "def " in content, f"No functions found in {file_path}"
        
        # Check for proper docstrings on key files
        if "train_model" in file_path or "serve" in file_path:
            lines = content.split('\n')
            has_docstring = any('"""' in line for line in lines[:20])
            assert has_docstring, f"Missing module docstring in {file_path}"
    
    print("✅ Code quality checks passed")

if __name__ == "__main__":
    print("🧪 Running self-tests for Modern ML Cookiecutter...")
    
    try:
        test_cookiecutter_structure()
        test_config_simplicity()
        test_dependencies() 
        test_tracelet_integration()
        test_code_quality()
        
        print("\n🎉 All tests passed! The template is ready for ML researchers.")
        print("\n📋 Quick Start:")
        print("1. cookiecutter https://github.com/your-repo/cookiecutter-modern-ml")
        print("2. cd your_project && uv sync")
        print("3. uv run task train")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        exit(1)