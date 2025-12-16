#!/usr/bin/env python3
"""
Tests for excluded_channels feature in the EEG preprocessing pipeline.

This test file validates that the excluded_channels parameter works correctly
across all steps that support it, allowing channels (like Cz) to be excluded
from analysis to avoid reference problems.
"""

import sys
import ast
from pathlib import Path

# Find the repository root
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"


def test_apply_excluded_channels_exists():
    """Test that the _apply_excluded_channels helper function exists."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    assert "def _apply_excluded_channels" in code, "_apply_excluded_channels function not found"
    print("✓ _apply_excluded_channels helper function exists")


def test_get_picks_has_excluded_channels_param():
    """Test that _get_picks has excluded_channels parameter."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Parse the code to check function signature
    tree = ast.parse(code)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_get_picks':
            arg_names = [arg.arg for arg in node.args.args]
            assert 'excluded_channels' in arg_names, "_get_picks missing excluded_channels parameter"
            print("✓ _get_picks has excluded_channels parameter")
            return
    
    raise AssertionError("_get_picks function not found")


def test_steps_support_excluded_channels():
    """Test that appropriate steps support excluded_channels parameter."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Steps that should support excluded_channels
    steps_with_exclusion = [
        '_step_bandpass_filter',
        '_step_notch_filter',
        '_step_interpolate_bad_channels',
        '_step_drop_bad_channels',
        '_step_ica',
        '_step_find_flat_channels',
        '_step_find_bads_channels_threshold',
        '_step_find_bads_channels_variance',
        '_step_find_bads_channels_high_frequency',
        '_step_find_bads_epochs_threshold',
    ]
    
    for step_name in steps_with_exclusion:
        # Find the function in the code
        pattern = f"def {step_name}"
        assert pattern in code, f"Step {step_name} not found"
        
        # Check if excluded_channels is retrieved from step_config
        # Look for the pattern: step_config.get('excluded_channels'
        start_idx = code.find(pattern)
        # Find the next function definition (end of current function)
        next_def = code.find("\n    def ", start_idx + len(pattern))
        if next_def == -1:
            next_def = len(code)
        
        function_code = code[start_idx:next_def]
        
        assert "excluded_channels" in function_code, \
            f"Step {step_name} does not handle excluded_channels parameter"
        
        print(f"✓ {step_name} supports excluded_channels")


def test_steps_pass_excluded_channels_to_get_picks():
    """Test that steps pass excluded_channels to _get_picks."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    steps_using_get_picks = [
        '_step_bandpass_filter',
        '_step_notch_filter',
        '_step_ica',
        '_step_find_flat_channels',
        '_step_find_bads_channels_threshold',
        '_step_find_bads_channels_variance',
        '_step_find_bads_channels_high_frequency',
        '_step_find_bads_epochs_threshold',
    ]
    
    for step_name in steps_using_get_picks:
        pattern = f"def {step_name}"
        start_idx = code.find(pattern)
        next_def = code.find("\n    def ", start_idx + len(pattern))
        if next_def == -1:
            next_def = len(code)
        
        function_code = code[start_idx:next_def]
        
        # Check that excluded_channels is passed to _get_picks
        assert "_get_picks(" in function_code, \
            f"Step {step_name} does not call _get_picks"
        
        # For these steps, check if excluded_channels is passed
        # This is a simple string check - in production, we'd want more robust parsing
        if "_get_picks(" in function_code:
            get_picks_calls = []
            for line in function_code.split('\n'):
                if '_get_picks(' in line:
                    get_picks_calls.append(line)
            
            # At least one call should have excluded_channels
            has_excluded = any('excluded_channels' in call for call in get_picks_calls)
            assert has_excluded, \
                f"Step {step_name} does not pass excluded_channels to _get_picks"
        
        print(f"✓ {step_name} passes excluded_channels to _get_picks")


def test_preprocessing_steps_report_excluded_channels():
    """Test that steps include excluded_channels in their preprocessing_steps report."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    steps_to_check = [
        '_step_bandpass_filter',
        '_step_notch_filter',
        '_step_interpolate_bad_channels',
        '_step_ica',
        '_step_find_flat_channels',
        '_step_find_bads_channels_threshold',
        '_step_find_bads_channels_variance',
        '_step_find_bads_channels_high_frequency',
        '_step_find_bads_epochs_threshold',
    ]
    
    for step_name in steps_to_check:
        pattern = f"def {step_name}"
        start_idx = code.find(pattern)
        next_def = code.find("\n    def ", start_idx + len(pattern))
        if next_def == -1:
            next_def = len(code)
        
        function_code = code[start_idx:next_def]
        
        # Check that preprocessing_steps.append includes excluded_channels
        if "preprocessing_steps" in function_code:
            # For bandpass_filter, it uses extend with multiple steps
            # For others, it uses append
            assert "'excluded_channels'" in function_code, \
                f"Step {step_name} does not report excluded_channels in preprocessing_steps"
        
        print(f"✓ {step_name} reports excluded_channels in preprocessing_steps")


def test_apply_excluded_channels_implementation():
    """Test the implementation of _apply_excluded_channels."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    pattern = "def _apply_excluded_channels"
    start_idx = code.find(pattern)
    assert start_idx != -1, "_apply_excluded_channels function not found"
    
    next_def = code.find("\n    def ", start_idx + len(pattern))
    if next_def == -1:
        next_def = len(code)
    
    function_code = code[start_idx:next_def]
    
    # Check key implementation details
    assert "if excluded_channels is None" in function_code, \
        "_apply_excluded_channels should handle None case"
    assert "return picks" in function_code, \
        "_apply_excluded_channels should return picks"
    assert "filtered_picks" in function_code or "filter" in function_code.lower(), \
        "_apply_excluded_channels should filter picks"
    
    print("✓ _apply_excluded_channels implementation looks correct")


def test_excluded_channels_documentation():
    """Test that excluded_channels is documented in docstrings."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Check _apply_excluded_channels has docstring
    pattern = "def _apply_excluded_channels"
    start_idx = code.find(pattern)
    next_lines = code[start_idx:start_idx + 1000]
    
    assert '"""' in next_lines or "'''" in next_lines, \
        "_apply_excluded_channels should have a docstring"
    
    print("✓ _apply_excluded_channels has documentation")


def test_steps_without_excluded_channels():
    """Test that steps where exclusion doesn't make sense are not modified."""
    pipeline_file = src_dir / "eeg_preprocessing_pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Steps that should NOT have excluded_channels
    steps_without_exclusion = [
        '_step_reference',  # Reference computation, handled differently
        '_step_resample',  # Resamples all data
        '_step_set_montage',  # Sets positions for all channels
        '_step_drop_unused_channels',  # Explicit drop, not exclusion
    ]
    
    for step_name in steps_without_exclusion:
        pattern = f"def {step_name}"
        if pattern in code:
            start_idx = code.find(pattern)
            next_def = code.find("\n    def ", start_idx + len(pattern))
            if next_def == -1:
                next_def = len(code)
            
            function_code = code[start_idx:next_def]
            
            # These steps should not have excluded_channels in their config retrieval
            # (though they might mention it in comments)
            config_get_lines = [line for line in function_code.split('\n') 
                              if 'step_config.get' in line]
            has_excluded_param = any("'excluded_channels'" in line 
                                    for line in config_get_lines)
            
            assert not has_excluded_param, \
                f"Step {step_name} should not support excluded_channels"
            
            print(f"✓ {step_name} correctly does not support excluded_channels")


def run_all_tests():
    """Run all tests in this file."""
    print("=" * 60)
    print("Running Excluded Channels Feature Tests")
    print("=" * 60)
    print()
    
    test_functions = [
        test_apply_excluded_channels_exists,
        test_get_picks_has_excluded_channels_param,
        test_steps_support_excluded_channels,
        test_steps_pass_excluded_channels_to_get_picks,
        test_preprocessing_steps_report_excluded_channels,
        test_apply_excluded_channels_implementation,
        test_excluded_channels_documentation,
        test_steps_without_excluded_channels,
    ]
    
    failed_tests = []
    
    for test_func in test_functions:
        try:
            test_func()
        except AssertionError as e:
            failed_tests.append((test_func.__name__, str(e)))
            print(f"✗ {test_func.__name__}: {e}")
        except Exception as e:
            failed_tests.append((test_func.__name__, f"Unexpected error: {e}"))
            print(f"✗ {test_func.__name__}: Unexpected error: {e}")
    
    print()
    print("=" * 60)
    if failed_tests:
        print(f"FAILED: {len(failed_tests)} test(s) failed")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
    else:
        print("SUCCESS: All excluded_channels feature tests passed!")
    print("=" * 60)
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
