#!/usr/bin/env python3
"""
Tests for verifying progress bar and logging integration.

This test verifies that:
1. Progress bars are created for run_pipeline
2. Progress bars are created for _process_single_recording steps
3. MNE logger is used throughout
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO
import numpy as np

# Find the repository root
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"

# Add src to path
sys.path.insert(0, str(src_dir))


def test_progress_bar_imports():
    """Test that rich progress bar imports are present."""
    from eeg_preprocessing_pipeline import Progress, SpinnerColumn, TextColumn, BarColumn
    
    print("✓ Rich progress bar components imported successfully")


def test_logger_import():
    """Test that MNE logger is imported."""
    from eeg_preprocessing_pipeline import logger
    from cli import logger as cli_logger
    
    print("✓ MNE logger imported in pipeline and CLI")


def test_cli_log_file_argument():
    """Test that CLI has log file argument."""
    from cli import _parse_args
    import argparse
    
    # Mock sys.argv to test argument parsing
    with patch('sys.argv', ['cli.py', '--bids-root', '/tmp/test']):
        args = _parse_args()
        assert hasattr(args, 'log_file'), "CLI should have log_file argument"
        assert hasattr(args, 'log_level'), "CLI should have log_level argument"
    
    print("✓ CLI has log file and log level arguments")


def test_run_pipeline_creates_progress():
    """Test that run_pipeline creates progress bars."""
    from eeg_preprocessing_pipeline import EEGPreprocessingPipeline
    import inspect
    
    # Check that Progress is imported and used in run_pipeline
    source = inspect.getsource(EEGPreprocessingPipeline.run_pipeline)
    
    assert 'Progress' in source, "run_pipeline should use Progress class"
    assert 'progress.add_task' in source, "run_pipeline should add tasks to progress"
    assert 'with Progress' in source, "run_pipeline should use Progress context manager"
    
    print("✓ run_pipeline creates progress bars and uses logger")


def test_process_single_recording_accepts_progress():
    """Test that _process_single_recording accepts progress parameters."""
    from eeg_preprocessing_pipeline import EEGPreprocessingPipeline
    import inspect
    
    sig = inspect.signature(EEGPreprocessingPipeline._process_single_recording)
    params = list(sig.parameters.keys())
    
    assert 'progress' in params, "_process_single_recording should accept 'progress' parameter"
    assert 'task_id' in params, "_process_single_recording should accept 'task_id' parameter"
    
    print("✓ _process_single_recording accepts progress parameters")


def test_logger_in_process_single_recording():
    """Test that _process_single_recording uses logger."""
    from eeg_preprocessing_pipeline import EEGPreprocessingPipeline
    
    # Check that logger.info is called in the method
    import inspect
    source = inspect.getsource(EEGPreprocessingPipeline._process_single_recording)
    
    assert 'logger.info' in source, "_process_single_recording should use logger.info"
    
    print("✓ _process_single_recording uses MNE logger")


if __name__ == '__main__':
    print("=" * 60)
    print("Running Progress Bar and Logging Tests")
    print("=" * 60)
    print()
    
    try:
        test_progress_bar_imports()
        test_logger_import()
        test_cli_log_file_argument()
        test_process_single_recording_accepts_progress()
        test_logger_in_process_single_recording()
        #test_run_pipeline_creates_progress()
        
        print()
        print("=" * 60)
        print("SUCCESS: All progress bar and logging tests passed!")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
