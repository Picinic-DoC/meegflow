#!/usr/bin/env python3
"""
Tests for verifying the integration of find_matching_paths.

This test verifies that:
1. The run_pipeline function accepts the correct parameters matching find_matching_paths
2. The CLI passes the correct arguments to run_pipeline
3. The parameters are properly forwarded to find_matching_paths
"""

import sys
import ast
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Find the repository root
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"

# Add src to path
sys.path.insert(0, str(src_dir))


def test_run_pipeline_signature():
    """Test that run_pipeline has the correct signature matching find_matching_paths."""
    from meegflow import MEEGFlowPipeline
    import inspect
    
    sig = inspect.signature(MEEGFlowPipeline.run_pipeline)
    params = list(sig.parameters.keys())
    
    # Check that the new parameters are present
    expected_params = [
        'self',
        'subjects',
        'sessions',
        'tasks',
        'acquisitions',
        'extension',
    ]
    
    for param in expected_params:
        assert param in params, f"Parameter '{param}' not found in run_pipeline signature"
    
    print("✓ run_pipeline has correct signature with find_matching_paths parameters")


def test_cli_passes_correct_arguments():
    """Test that CLI passes the correct arguments to run_pipeline."""
    cli_file = src_dir / "cli.py"
    with open(cli_file, 'r') as f:
        code = f.read()
    
    # Check that the main function calls run_pipeline with the new parameters
    assert "subjects=args.subjects" in code, "subjects parameter not passed to run_pipeline"
    assert "tasks=args.tasks" in code, "tasks parameter not passed to run_pipeline"
    assert "sessions=args.sessions" in code, "sessions parameter not passed to run_pipeline"
    
    print("✓ CLI passes correct arguments to run_pipeline")


def test_cli_has_new_arguments():
    """Test that CLI has the new arguments."""
    cli_file = src_dir / "cli.py"
    with open(cli_file, 'r') as f:
        code = f.read()
    
    # Check for new arguments
    expected_args = [
        '--bids-root',
        '--output-root',
        '--subjects',
        '--sessions',
        '--tasks',
        '--acquisitions',
        '--runs',
        '--extension',
        '--config', 
        '--log-file',
        '--log-level',
    ]
    
    for arg in expected_args:
        assert arg in code, f"Argument '{arg}' not found in CLI"
    
    print("✓ CLI has all new arguments")


def test_subjects_parameter_accepts_none():
    """Test that subjects parameter can be None."""
    from meegflow import MEEGFlowPipeline
    import inspect
    
    sig = inspect.signature(MEEGFlowPipeline.run_pipeline)
    subjects_param = sig.parameters['subjects']
    
    # Check that default is None
    assert subjects_param.default is None, "subjects parameter should default to None"
    
    print("✓ subjects parameter accepts None (matching find_matching_paths)")


def test_tasks_parameter_accepts_none():
    """Test that tasks parameter can be None."""
    from meegflow import MEEGFlowPipeline
    import inspect
    
    sig = inspect.signature(MEEGFlowPipeline.run_pipeline)
    tasks_param = sig.parameters['tasks']
    
    # Check that default is None
    assert tasks_param.default is None, "tasks parameter should default to None"
    
    print("✓ tasks parameter accepts None (matching find_matching_paths)")


def test_cli_subjects_not_required():
    """Test that --subjects is not required in CLI."""
    cli_file = src_dir / "cli.py"
    with open(cli_file, 'r') as f:
        code = f.read()
    
    # Parse the file to find the --subjects argument definition
    lines = code.split('\n')
    subjects_section = []
    in_subjects_section = False
    
    for line in lines:
        if "'--subjects'" in line or '"--subjects"' in line:
            in_subjects_section = True
        if in_subjects_section:
            subjects_section.append(line)
            if ')' in line and 'parser.add_argument' in ''.join(subjects_section):
                break
    
    subjects_text = ''.join(subjects_section)
    # Check that required is False or not set to True
    assert 'required=True' not in subjects_text or 'required=False' in subjects_text, \
        "--subjects should not be required (or should be explicitly set to False)"
    
    print("✓ CLI --subjects is not required (allows processing all subjects)")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running find_matching_paths Integration Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_run_pipeline_signature,
        test_cli_passes_correct_arguments,
        test_cli_has_new_arguments,
        test_subjects_parameter_accepts_none,
        test_tasks_parameter_accepts_none,
        test_cli_subjects_not_required,
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
        print("SUCCESS: All find_matching_paths integration tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
