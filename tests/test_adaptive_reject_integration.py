#!/usr/bin/env python3
"""
Tests for adaptive autoreject integration in MEEG preprocessing pipeline.

These are structural tests that verify the integration without requiring MNE.
"""

import sys
import ast
import json
import yaml
from pathlib import Path


def test_adaptive_reject_import():
    """Test that adaptive_reject is imported in the pipeline."""
    pipeline_file = Path("src/meegflow.py")
    assert pipeline_file.exists(), "Pipeline file does not exist"
    
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    assert "import adaptive_reject" in code, "adaptive_reject not imported"
    print("✓ adaptive_reject module is imported")


def test_step_functions_registered():
    """Test that all adaptive autoreject step functions are registered."""
    pipeline_file = Path("src/meegflow.py")
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    required_steps = [
        'find_bads_channels_threshold',
        'find_bads_channels_variance',
        'find_bads_channels_high_frequency',
        'find_bads_epochs_threshold'
    ]
    
    for step in required_steps:
        # Check if step is in the step_functions dictionary
        expected = f"'{step}': self._step_{step}"
        assert expected in code, f"Step {step} not registered in step_functions"
        print(f"✓ Step '{step}' is registered")


def test_step_methods_defined():
    """Test that all adaptive autoreject step methods are defined."""
    pipeline_file = Path("src/meegflow.py")
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    required_methods = [
        '_step_find_bads_channels_threshold',
        '_step_find_bads_channels_variance',
        '_step_find_bads_channels_high_frequency',
        '_step_find_bads_epochs_threshold'
    ]
    
    for method in required_methods:
        # Check if method is defined
        pattern = f"def {method}(self, data: Dict[str, Any], step_config: Dict[str, Any])"
        assert pattern in code, f"Method {method} not defined with correct signature"
        print(f"✓ Method '{method}' is defined")


def test_step_methods_call_adaptive_reject():
    """Test that step methods call the corresponding adaptive_reject functions."""
    pipeline_file = Path("src/meegflow.py")
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    function_calls = [
        ('_step_find_bads_channels_threshold', 'adaptive_reject.find_bads_channels_threshold'),
        ('_step_find_bads_channels_variance', 'adaptive_reject.find_bads_channels_variance'),
        ('_step_find_bads_channels_high_frequency', 'adaptive_reject.find_bads_channels_high_frequency'),
        ('_step_find_bads_epochs_threshold', 'adaptive_reject.find_bads_epochs_threshold')
    ]
    
    for method, call in function_calls:
        assert call in code, f"Method {method} does not call {call}"
        print(f"✓ Method '{method}' calls '{call}'")


def test_readme_documents_adaptive_reject():
    """Test that README documents the adaptive reject steps."""
    readme_file = Path("README.md")
    assert readme_file.exists(), "README.md does not exist"
    
    with open(readme_file, 'r') as f:
        readme = f.read()
    
    required_steps = [
        'find_bads_channels_threshold',
        'find_bads_channels_variance',
        'find_bads_channels_high_frequency',
        'find_bads_epochs_threshold'
    ]
    
    for step in required_steps:
        assert step in readme, f"Step '{step}' not documented in README"
        print(f"✓ README documents step '{step}'")


def test_readme_has_config_example():
    """Test that README references the new config example."""
    readme_file = Path("README.md")
    
    with open(readme_file, 'r') as f:
        readme = f.read()
    
    assert "config_with_adaptive_reject.yaml" in readme, \
        "README does not reference config_with_adaptive_reject.yaml"
    print("✓ README references 'config_with_adaptive_reject.yaml'")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Adaptive Autoreject Integration Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_adaptive_reject_import,
        test_step_functions_registered,
        test_step_methods_defined,
        test_step_methods_call_adaptive_reject,
        test_readme_documents_adaptive_reject,
        test_readme_has_config_example,
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
        print("SUCCESS: All adaptive autoreject integration tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
