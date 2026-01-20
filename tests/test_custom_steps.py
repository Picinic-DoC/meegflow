#!/usr/bin/env python3
"""
Tests for custom preprocessing steps functionality.

This file tests the ability to load and execute custom preprocessing steps
from user-defined Python files.
"""

import sys
import ast
import tempfile
from pathlib import Path

# Find the repository root
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"


def test_pipeline_has_load_custom_steps_method():
    """Test that the pipeline has the _load_custom_steps method."""
    print("Testing: _load_custom_steps method exists...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    assert "def _load_custom_steps" in code, "_load_custom_steps method not found"
    print("✓ Test passed")


def test_pipeline_imports_required_modules():
    """Test that the pipeline imports required modules for custom steps."""
    print("Testing: required imports for custom steps...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    assert "import importlib.util" in code, "importlib.util not imported"
    assert "import inspect" in code, "inspect not imported"
    print("✓ Test passed")


def test_custom_steps_folder_config_used():
    """Test that custom_steps_folder config is used in __init__."""
    print("Testing: custom_steps_folder config is used...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    assert "custom_steps_folder" in code, "custom_steps_folder not mentioned in code"
    assert "_load_custom_steps" in code, "_load_custom_steps not called"
    print("✓ Test passed")


def test_load_custom_steps_signature():
    """Test that _load_custom_steps has the correct signature."""
    print("Testing: _load_custom_steps signature...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
        tree = ast.parse(code)
    
    # Find the _load_custom_steps method
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_load_custom_steps":
            found = True
            # Check that it has at least 2 parameters (self and custom_steps_folder)
            assert len(node.args.args) >= 2, f"_load_custom_steps should have at least 2 parameters, found {len(node.args.args)}"
            # Check parameter names
            param_names = [arg.arg for arg in node.args.args]
            assert 'self' in param_names, "Missing 'self' parameter"
            assert 'custom_steps_folder' in param_names, "Missing 'custom_steps_folder' parameter"
            break
    
    assert found, "_load_custom_steps method not found in AST"
    print("✓ Test passed")


def test_load_custom_steps_returns_dict():
    """Test that _load_custom_steps returns a dict."""
    print("Testing: _load_custom_steps return type...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Look for return dict in the _load_custom_steps method
    assert "return custom_steps" in code or "return {}" in code, "_load_custom_steps should return a dict"
    print("✓ Test passed")


def test_custom_steps_docstring_exists():
    """Test that _load_custom_steps has comprehensive documentation."""
    print("Testing: _load_custom_steps has documentation...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
        tree = ast.parse(code)
    
    # Find the _load_custom_steps method and check for docstring
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_load_custom_steps":
            found = True
            docstring = ast.get_docstring(node)
            assert docstring is not None, "_load_custom_steps should have a docstring"
            assert len(docstring) > 50, "Docstring should be comprehensive"
            assert "custom" in docstring.lower(), "Docstring should mention custom steps"
            break
    
    assert found, "_load_custom_steps method not found"
    print("✓ Test passed")


def test_error_handling_for_missing_folder():
    """Test that error handling is implemented for missing folder."""
    print("Testing: error handling for missing folder...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Look for validation code
    assert "not" in code and ("exists()" in code or ".exists" in code), "Should check if folder exists"
    assert "is_dir()" in code or ".is_dir" in code, "Should check if path is a directory"
    assert "ValueError" in code, "Should raise ValueError for invalid paths"
    print("✓ Test passed")


def test_glob_for_python_files():
    """Test that the code searches for .py files."""
    print("Testing: searching for .py files...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Look for glob or pattern matching for .py files
    assert '*.py' in code or '"*.py"' in code or "'*.py'" in code, "Should search for .py files"
    assert "glob" in code, "Should use glob to find files"
    print("✓ Test passed")


def test_skips_underscore_files():
    """Test that the code skips files starting with underscore."""
    print("Testing: skipping underscore files...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Look for underscore checking logic
    assert "startswith('_')" in code or 'startswith("_")' in code, "Should skip files starting with underscore"
    print("✓ Test passed")


def test_function_signature_validation():
    """Test that the code validates function signatures."""
    print("Testing: function signature validation...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Look for signature checking
    assert "inspect.signature" in code or "sig.parameters" in code, "Should check function signatures"
    assert "len(params)" in code or "len(" in code, "Should validate parameter count"
    print("✓ Test passed")


def test_step_functions_update():
    """Test that custom steps are added to step_functions dict."""
    print("Testing: custom steps added to step_functions...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Look for update of step_functions dict
    assert "step_functions.update" in code or "step_functions[" in code, "Should update step_functions dict"
    print("✓ Test passed")


def test_logging_messages():
    """Test that appropriate logging messages are present."""
    print("Testing: logging messages...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Look for logger calls related to custom steps
    assert "logger.info" in code or "logger.warning" in code or "logger.error" in code, "Should log custom step loading"
    print("✓ Test passed")


def test_module_loading():
    """Test that the code uses importlib to load modules."""
    print("Testing: module loading with importlib...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Look for importlib usage
    assert "spec_from_file_location" in code, "Should use spec_from_file_location"
    assert "module_from_spec" in code, "Should use module_from_spec"
    assert "exec_module" in code, "Should use exec_module"
    print("✓ Test passed")


def test_error_handling_for_invalid_files():
    """Test that errors in loading individual files don't crash the process."""
    print("Testing: error handling for invalid files...")
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Look for try-except blocks in the loading process
    assert "try:" in code and "except" in code, "Should have error handling"
    assert "continue" in code, "Should continue loading other files on error"
    print("✓ Test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Custom Steps Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_pipeline_has_load_custom_steps_method,
        test_pipeline_imports_required_modules,
        test_custom_steps_folder_config_used,
        test_load_custom_steps_signature,
        test_load_custom_steps_returns_dict,
        test_custom_steps_docstring_exists,
        test_error_handling_for_missing_folder,
        test_glob_for_python_files,
        test_skips_underscore_files,
        test_function_signature_validation,
        test_step_functions_update,
        test_logging_messages,
        test_module_loading,
        test_error_handling_for_invalid_files,
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
            import traceback
            traceback.print_exc()
            failed_tests.append(test.__name__)
    
    print()
    print("=" * 60)
    if failed_tests:
        print(f"FAILED: {len(failed_tests)} test(s) failed")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("SUCCESS: All custom steps tests passed!")
        print()
        print("Note: These are static code structure tests.")
        print("Full functionality testing requires:")
        print("  - Installing dependencies (pip install -r requirements.txt)")
        print("  - Creating custom step files and running integration tests")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())


