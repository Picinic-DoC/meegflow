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


def test_pipeline_file_exists():
    """Test that the main pipeline file exists."""
    pipeline_file = Path("eeg_preprocessing_pipeline.py")
    assert pipeline_file.exists(), "Pipeline file does not exist"
    print("✓ Pipeline file exists")


def test_pipeline_syntax():
    """Test that the pipeline file has valid Python syntax."""
    pipeline_file = Path("eeg_preprocessing_pipeline.py")
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    try:
        ast.parse(code)
        print("✓ Pipeline file has valid syntax")
    except SyntaxError as e:
        raise AssertionError(f"Syntax error in pipeline: {e}")


def test_pipeline_has_required_classes():
    """Test that the pipeline file contains required classes."""
    pipeline_file = Path("eeg_preprocessing_pipeline.py")
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    assert "class EEGPreprocessingPipeline" in code, "EEGPreprocessingPipeline class not found"
    print("✓ Required class EEGPreprocessingPipeline found")


def test_pipeline_has_required_methods():
    """Test that the pipeline class has required methods."""
    pipeline_file = Path("eeg_preprocessing_pipeline.py")
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    required_methods = [
        "read_data",
        "preprocess_raw",
        "apply_ica",
        "create_epochs",
        "save_epochs",
        "generate_html_report",
        "generate_json_report",
        "run_pipeline"
    ]
    
    for method in required_methods:
        assert f"def {method}" in code, f"Method {method} not found"
        print(f"✓ Method {method} found")


def test_config_example_valid_json():
    """Test that the example config is valid JSON."""
    config_file = Path("config_example.json")
    assert config_file.exists(), "Config example file does not exist"
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Check for required keys
    required_keys = [
        "l_freq", "h_freq", "epochs_tmin", "epochs_tmax",
        "baseline", "reject_criteria", "ica_n_components", "ica_method"
    ]
    
    for key in required_keys:
        assert key in config, f"Required config key {key} not found"
    
    print("✓ Config example is valid JSON with required keys")


def test_requirements_file_exists():
    """Test that requirements.txt exists and contains necessary packages."""
    req_file = Path("requirements.txt")
    assert req_file.exists(), "requirements.txt does not exist"
    
    with open(req_file, 'r') as f:
        requirements = f.read()
    
    required_packages = ["mne", "mne-bids", "numpy", "scipy"]
    
    for package in required_packages:
        assert package in requirements, f"Required package {package} not in requirements.txt"
    
    print("✓ requirements.txt exists with required packages")


def test_output_directories_structure():
    """Test that the pipeline creates correct output directory structure."""
    pipeline_file = Path("eeg_preprocessing_pipeline.py")
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Check that the three required output directories are mentioned
    assert "clean_epochs" in code, "clean_epochs directory not mentioned"
    assert "html_reports" in code, "html_reports directory not mentioned"
    assert "json_reports" in code, "json_reports directory not mentioned"
    
    print("✓ All three output directories are configured")


def test_readme_exists():
    """Test that README exists and mentions key features."""
    readme_file = Path("README.md")
    assert readme_file.exists(), "README.md does not exist"
    
    with open(readme_file, 'r') as f:
        readme = f.read()
    
    required_sections = [
        "MNE-BIDS",
        "clean_epochs",
        "html_reports",
        "json_reports",
        "Installation",
        "Usage"
    ]
    
    for section in required_sections:
        assert section in readme, f"Required section '{section}' not in README"
    
    print("✓ README.md exists with required sections")


def test_slurm_script_exists():
    """Test that SLURM script exists."""
    slurm_file = Path("run_slurm.sh")
    assert slurm_file.exists(), "run_slurm.sh does not exist"
    
    with open(slurm_file, 'r') as f:
        content = f.read()
    
    assert "#SBATCH" in content, "SLURM directives not found"
    assert "eeg_preprocessing_pipeline.py" in content, "Pipeline script not referenced"
    
    print("✓ SLURM script exists and is properly configured")


def test_example_usage_exists():
    """Test that example usage script exists."""
    example_file = Path("example_usage.py")
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
        test_config_example_valid_json,
        test_requirements_file_exists,
        test_output_directories_structure,
        test_readme_exists,
        test_slurm_script_exists,
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
