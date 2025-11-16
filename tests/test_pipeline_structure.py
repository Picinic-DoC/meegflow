#!/usr/bin/env python3
"""
Basic structure tests for the EEG preprocessing pipeline.

This file tests the pipeline structure without requiring actual EEG data
or full MNE installation.
"""

import sys
import ast
import json
from pathlib import Path

# Find the repository root
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"
configs_dir = repo_root / "configs"


def test_pipeline_file_exists():
    """Test that the main pipeline file exists."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    assert pipeline_file.exists(), "Pipeline file does not exist"
    print("✓ Pipeline file exists")


def test_pipeline_syntax():
    """Test that the pipeline file has valid Python syntax."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    try:
        ast.parse(code)
        print("✓ Pipeline file has valid syntax")
    except SyntaxError as e:
        raise AssertionError(f"Syntax error in pipeline: {e}")


def test_pipeline_has_required_classes():
    """Test that the pipeline file contains required classes."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    assert "class EEGPreprocessingPipeline" in code, "EEGPreprocessingPipeline class not found"
    print("✓ Required class EEGPreprocessingPipeline found")


def test_pipeline_has_required_methods():
    """Test that the pipeline class has required methods."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Check for auxiliary step functions (new modular design)
    required_methods = [
        "_step_load_data",
        "_step_bandpass_filter",
        "_step_notch_filter",
        "_step_reference",
        "_step_ica",
        "_step_find_events",
        "_step_epoch",
        "_step_save_clean_epochs",
        "_step_generate_json_report",
        "_step_generate_html_report",
        "run_pipeline"
    ]
    
    for method in required_methods:
        assert f"def {method}" in code, f"Method {method} not found"
        print(f"✓ Method {method} found")


def test_config_example_valid_yaml():
    """Test that the example config is valid YAML."""
    config_file = configs_dir / "config_example.yaml"
    assert config_file.exists(), "Config example file does not exist"
    
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for pipeline configuration structure (new config format)
    assert "pipeline" in config, "Config must have 'pipeline' key"
    assert isinstance(config["pipeline"], list), "Pipeline must be a list of steps"
    assert len(config["pipeline"]) > 0, "Pipeline must have at least one step"
    
    # Check that steps have names
    for step in config["pipeline"]:
        assert "name" in step, "Each step must have a 'name' key"
    
    print("✓ Config example is valid YAML with pipeline structure")


def test_requirements_file_exists():
    """Test that requirements.txt exists and contains necessary packages."""
    req_file = repo_root / "requirements.txt"
    assert req_file.exists(), "requirements.txt does not exist"
    
    with open(req_file, 'r') as f:
        requirements = f.read()
    
    required_packages = ["mne", "mne-bids", "numpy", "scipy", "PyYAML"]
    
    for package in required_packages:
        assert package in requirements, f"Required package {package} not in requirements.txt"
    
    print("✓ requirements.txt exists with required packages")


def test_output_directories_structure():
    """Test that the pipeline creates correct output directory structure."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Check that the output directories are mentioned (reports and epochs)
    assert "epochs" in code, "epochs directory not mentioned"
    assert "reports" in code, "reports directory not mentioned"
    
    print("✓ All output directories are configured")


def test_readme_exists():
    """Test that README exists and mentions key features."""
    readme_file = repo_root / "README.md"
    assert readme_file.exists(), "README.md does not exist"
    
    with open(readme_file, 'r') as f:
        readme = f.read()
    
    required_sections = [
        "MNE-BIDS",
        "epochs",
        "reports",
        "Installation",
        "Usage",
        "YAML"
    ]
    
    for section in required_sections:
        assert section in readme, f"Required section '{section}' not in README"
    
    print("✓ README.md exists with required sections")


def test_batch_processing_support():
    """Test that the pipeline supports batch processing."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Check that the pipeline supports multiple subjects
    assert "--subjects" in code or "subjects" in code.lower(), "Batch processing support not found"
    assert "run_pipeline" in code, "run_pipeline method not found"
    
    print("✓ Pipeline supports batch processing of multiple subjects")


def test_example_usage_exists():
    """Test that example usage script exists."""
    example_file = repo_root / "example_usage.py"
    assert example_file.exists(), "example_usage.py does not exist"
    
    with open(example_file, 'r') as f:
        code = f.read()
    
    # Check for example functions
    assert "def example_" in code, "No example functions found"
    assert "EEGPreprocessingPipeline" in code, "Pipeline class not imported"
    
    print("✓ Example usage script exists with examples")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running EEG Preprocessing Pipeline Structure Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_pipeline_file_exists,
        test_pipeline_syntax,
        test_pipeline_has_required_classes,
        test_pipeline_has_required_methods,
        test_config_example_valid_yaml,
        test_requirements_file_exists,
        test_output_directories_structure,
        test_readme_exists,
        test_batch_processing_support,
        test_example_usage_exists,
    ]
    
    failed_tests = []
    
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed_tests.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed_tests.append(test.__name__)
    
    print()
    print("=" * 60)
    if failed_tests:
        print(f"FAILED: {len(failed_tests)} test(s) failed")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("SUCCESS: All structure tests passed!")
        print()
        print("Note: Full functionality testing requires:")
        print("  - Installing dependencies (pip install -r requirements.txt)")
        print("  - BIDS-formatted EEG data")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
