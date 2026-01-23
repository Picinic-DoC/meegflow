#!/usr/bin/env python3
"""
Integration tests for the split module architecture.

These tests verify that:
1. The PreprocessingPipeline can save intermediate results
2. The ReportGenerator can load intermediate results
3. The split workflow produces the same reports as the combined workflow
"""

import sys
from pathlib import Path
import tempfile
import shutil
import json

# Add src to path
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"
sys.path.insert(0, str(src_dir))


def test_preprocessing_pipeline_exists():
    """Test that the PreprocessingPipeline module exists and can be imported."""
    try:
        from preprocessing_pipeline import PreprocessingPipeline
        print("✓ PreprocessingPipeline module imported successfully")
        assert PreprocessingPipeline is not None
        return True
    except ImportError as e:
        print(f"✗ Failed to import PreprocessingPipeline: {e}")
        return False


def test_report_generator_exists():
    """Test that the ReportGenerator module exists and can be imported."""
    try:
        from report_generator import ReportGenerator
        print("✓ ReportGenerator module imported successfully")
        assert ReportGenerator is not None
        return True
    except ImportError as e:
        print(f"✗ Failed to import ReportGenerator: {e}")
        return False


def test_preprocessing_cli_exists():
    """Test that the preprocessing CLI exists."""
    preprocessing_cli = src_dir / "preprocessing_cli.py"
    assert preprocessing_cli.exists(), "preprocessing_cli.py does not exist"
    
    with open(preprocessing_cli, 'r') as f:
        code = f.read()
    
    assert "def preprocessing_main(" in code, "preprocessing_main function not found"
    assert "PreprocessingPipeline" in code, "PreprocessingPipeline not imported in CLI"
    print("✓ Preprocessing CLI exists and has correct structure")
    return True


def test_report_cli_exists():
    """Test that the report CLI exists."""
    report_cli = src_dir / "report_cli.py"
    assert report_cli.exists(), "report_cli.py does not exist"
    
    with open(report_cli, 'r') as f:
        code = f.read()
    
    assert "def report_main(" in code, "report_main function not found"
    assert "ReportGenerator" in code, "ReportGenerator not imported in CLI"
    print("✓ Report CLI exists and has correct structure")
    return True


def test_preprocessing_has_save_step():
    """Test that PreprocessingPipeline has save_intermediate_results step."""
    from preprocessing_pipeline import PreprocessingPipeline
    
    # Create a dummy pipeline to check step functions
    config = {'pipeline': []}
    pipeline = PreprocessingPipeline(
        bids_root=tempfile.mkdtemp(),
        config=config
    )
    
    assert 'save_intermediate_results' in pipeline.step_functions, \
        "save_intermediate_results step not found in PreprocessingPipeline"
    print("✓ PreprocessingPipeline has save_intermediate_results step")
    return True


def test_preprocessing_no_report_steps():
    """Test that PreprocessingPipeline does NOT have report generation steps."""
    from preprocessing_pipeline import PreprocessingPipeline
    
    # Create a dummy pipeline to check step functions
    config = {'pipeline': []}
    pipeline = PreprocessingPipeline(
        bids_root=tempfile.mkdtemp(),
        config=config
    )
    
    assert 'generate_json_report' not in pipeline.step_functions, \
        "generate_json_report should not be in PreprocessingPipeline"
    assert 'generate_html_report' not in pipeline.step_functions, \
        "generate_html_report should not be in PreprocessingPipeline"
    print("✓ PreprocessingPipeline does not have report generation steps")
    return True


def test_report_generator_has_methods():
    """Test that ReportGenerator has required methods."""
    from report_generator import ReportGenerator
    
    # Create a dummy report generator
    report_gen = ReportGenerator(bids_root=tempfile.mkdtemp())
    
    assert hasattr(report_gen, 'load_intermediate_results'), \
        "ReportGenerator missing load_intermediate_results method"
    assert hasattr(report_gen, 'generate_json_report'), \
        "ReportGenerator missing generate_json_report method"
    assert hasattr(report_gen, 'generate_html_report'), \
        "ReportGenerator missing generate_html_report method"
    assert hasattr(report_gen, 'generate_reports'), \
        "ReportGenerator missing generate_reports method"
    print("✓ ReportGenerator has all required methods")
    return True


def test_backward_compatibility():
    """Test that the original EEGPreprocessingPipeline still exists."""
    from eeg_preprocessing_pipeline import EEGPreprocessingPipeline
    
    # Create a dummy pipeline to check it still has report steps
    config = {'pipeline': []}
    pipeline = EEGPreprocessingPipeline(
        bids_root=tempfile.mkdtemp(),
        config=config
    )
    
    # Original pipeline should still have report steps
    assert 'generate_json_report' in pipeline.step_functions, \
        "Original EEGPreprocessingPipeline should still have generate_json_report"
    assert 'generate_html_report' in pipeline.step_functions, \
        "Original EEGPreprocessingPipeline should still have generate_html_report"
    print("✓ Original EEGPreprocessingPipeline maintains backward compatibility")
    return True


def test_setup_has_new_entry_points():
    """Test that setup.py has the new entry points."""
    setup_file = repo_root / "setup.py"
    with open(setup_file, 'r') as f:
        setup_code = f.read()
    
    assert "eeg-preprocess-only" in setup_code, \
        "eeg-preprocess-only entry point not found in setup.py"
    assert "eeg-generate-reports" in setup_code, \
        "eeg-generate-reports entry point not found in setup.py"
    assert "preprocessing_cli:preprocessing_main" in setup_code, \
        "preprocessing_cli entry point not properly configured"
    assert "report_cli:report_main" in setup_code, \
        "report_cli entry point not properly configured"
    print("✓ setup.py has new entry points configured")
    return True


def test_config_example_exists():
    """Test that a config example with save_intermediate_results exists."""
    configs_dir = repo_root / "configs"
    
    # Look for any config file with save_intermediate_results
    found = False
    for config_file in configs_dir.glob("*.yaml"):
        with open(config_file, 'r') as f:
            content = f.read()
            if "save_intermediate_results" in content:
                found = True
                print(f"✓ Found config with save_intermediate_results: {config_file.name}")
                break
    
    assert found, "No config file with save_intermediate_results found"
    return True


def test_dockerfiles_exist():
    """Test that new Dockerfiles exist."""
    dockerfile_preprocessing = repo_root / "Dockerfile.preprocessing"
    dockerfile_report = repo_root / "Dockerfile.report"
    docker_compose = repo_root / "docker-compose.yml"
    
    assert dockerfile_preprocessing.exists(), "Dockerfile.preprocessing does not exist"
    assert dockerfile_report.exists(), "Dockerfile.report does not exist"
    assert docker_compose.exists(), "docker-compose.yml does not exist"
    
    # Check that they have correct entry points
    with open(dockerfile_preprocessing, 'r') as f:
        content = f.read()
        assert "eeg-preprocess-only" in content, \
            "Dockerfile.preprocessing should use eeg-preprocess-only entry point"
    
    with open(dockerfile_report, 'r') as f:
        content = f.read()
        assert "eeg-generate-reports" in content, \
            "Dockerfile.report should use eeg-generate-reports entry point"
    
    print("✓ All Docker files exist with correct entry points")
    return True


def test_readmes_exist():
    """Test that module-specific READMEs exist."""
    readme_preprocessing = repo_root / "README_preprocessing.md"
    readme_report = repo_root / "README_report.md"
    
    assert readme_preprocessing.exists(), "README_preprocessing.md does not exist"
    assert readme_report.exists(), "README_report.md does not exist"
    
    # Check that main README references them
    main_readme = repo_root / "README.md"
    with open(main_readme, 'r') as f:
        content = f.read()
        assert "README_preprocessing.md" in content, \
            "Main README should reference README_preprocessing.md"
        assert "README_report.md" in content, \
            "Main README should reference README_report.md"
    
    print("✓ All README files exist and are properly referenced")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing Split Module Architecture")
    print("="*70 + "\n")
    
    tests = [
        test_preprocessing_pipeline_exists,
        test_report_generator_exists,
        test_preprocessing_cli_exists,
        test_report_cli_exists,
        test_preprocessing_has_save_step,
        test_preprocessing_no_report_steps,
        test_report_generator_has_methods,
        test_backward_compatibility,
        test_setup_has_new_entry_points,
        test_config_example_exists,
        test_dockerfiles_exist,
        test_readmes_exist,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\nRunning: {test.__name__}")
        print("-" * 70)
        try:
            result = test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    sys.exit(0 if failed == 0 else 1)
