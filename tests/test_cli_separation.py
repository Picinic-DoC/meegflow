#!/usr/bin/env python3
"""
Tests for verifying the separation of CLI and API.

This test verifies that:
1. The EEGPreprocessingPipeline class can be imported for API usage
2. The CLI module exists and has the correct structure
3. The CLI and API are properly separated
"""

import sys
import ast
from pathlib import Path

# Find the repository root
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"


def test_cli_file_exists():
    """Test that the CLI file exists."""
    cli_file = src_dir / "cli.py"
    assert cli_file.exists(), "CLI file does not exist"
    print("✓ CLI file exists")


def test_cli_has_main_function():
    """Test that the CLI file has a main function."""
    cli_file = src_dir / "cli.py"
    with open(cli_file, 'r') as f:
        code = f.read()
    
    assert "def main(" in code, "main function not found in CLI"
    assert "if __name__ == '__main__':" in code, "Main execution block not found in CLI"
    print("✓ CLI has main function and execution block")


def test_cli_has_arg_parser():
    """Test that the CLI file has argument parser."""
    cli_file = src_dir / "cli.py"
    with open(cli_file, 'r') as f:
        code = f.read()
    
    assert "argparse" in code, "argparse not imported in CLI"
    assert "ArgumentParser" in code, "ArgumentParser not used in CLI"
    assert "--bids-root" in code, "--bids-root argument not found"
    assert "--subjects" in code, "--subjects argument not found"
    print("✓ CLI has argument parser with required arguments")


def test_cli_imports_pipeline():
    """Test that the CLI imports the EEGPreprocessingPipeline."""
    cli_file = src_dir / "cli.py"
    with open(cli_file, 'r') as f:
        code = f.read()
    
    assert "from eeg_preprocessing_pipeline import EEGPreprocessingPipeline" in code, \
        "CLI does not import EEGPreprocessingPipeline"
    print("✓ CLI imports EEGPreprocessingPipeline")


def test_pipeline_has_no_cli_code():
    """Test that the pipeline file does not contain CLI code."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Check that argparse is not imported
    assert "import argparse" not in code, "Pipeline file still imports argparse"
    
    # Check that there's no main function
    assert "def main(" not in code, "Pipeline file still has main function"
    
    # Check that there's no if __name__ == '__main__'
    assert "if __name__ == '__main__':" not in code, "Pipeline file still has main execution block"
    
    print("✓ Pipeline file does not contain CLI code")


def test_pipeline_has_class():
    """Test that the pipeline file still has the EEGPreprocessingPipeline class."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    assert "class EEGPreprocessingPipeline" in code, "EEGPreprocessingPipeline class not found"
    print("✓ Pipeline file has EEGPreprocessingPipeline class")


def test_setup_file_exists():
    """Test that setup.py exists for package installation."""
    setup_file = repo_root / "setup.py"
    assert setup_file.exists(), "setup.py does not exist"
    print("✓ setup.py exists")


def test_setup_has_entry_point():
    """Test that setup.py has CLI entry point."""
    setup_file = repo_root / "setup.py"
    with open(setup_file, 'r') as f:
        code = f.read()
    
    assert "entry_points" in code, "entry_points not found in setup.py"
    assert "console_scripts" in code, "console_scripts not found in setup.py"
    assert "cli:main" in code, "CLI entry point not found in setup.py"
    print("✓ setup.py has CLI entry point")


def test_api_import_structure():
    """Test that the API import structure is correct."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    
    with open(pipeline_file, 'r') as f:
        tree = ast.parse(f.read())
    
    # Check that EEGPreprocessingPipeline class exists
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert 'EEGPreprocessingPipeline' in classes, "EEGPreprocessingPipeline class not found in AST"
    
    # Check that run_pipeline method exists
    methods = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert 'run_pipeline' in methods, "run_pipeline method not found"
    
    print("✓ API import structure is correct")


def test_cli_import_structure():
    """Test that the CLI import structure is correct."""
    cli_file = src_dir / "cli.py"
    
    with open(cli_file, 'r') as f:
        tree = ast.parse(f.read())
    
    # Check that main and _parse_args functions exist
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert 'main' in functions, "main function not found in CLI AST"
    assert '_parse_args' in functions, "_parse_args function not found in CLI AST"
    
    print("✓ CLI import structure is correct")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running CLI/API Separation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_cli_file_exists,
        test_cli_has_main_function,
        test_cli_has_arg_parser,
        test_cli_imports_pipeline,
        test_pipeline_has_no_cli_code,
        test_pipeline_has_class,
        test_setup_file_exists,
        test_setup_has_entry_point,
        test_api_import_structure,
        test_cli_import_structure,
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
        print("SUCCESS: All CLI/API separation tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
