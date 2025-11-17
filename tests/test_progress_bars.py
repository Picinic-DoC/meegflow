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
    import mne
    
    # Mock the necessary components
    with patch('eeg_preprocessing_pipeline.find_matching_paths') as mock_find_paths, \
         patch('eeg_preprocessing_pipeline.read_raw_bids') as mock_read_raw, \
         patch('eeg_preprocessing_pipeline.logger') as mock_logger, \
         patch('eeg_preprocessing_pipeline.Progress') as mock_progress_class:
        
        # Setup mocks
        mock_bids_path = Mock()
        mock_bids_path.subject = '01'
        mock_bids_path.task = 'rest'
        mock_bids_path.session = None
        mock_bids_path.acquisition = None
        mock_bids_path.run = None
        mock_bids_path.basename = 'sub-01_task-rest_eeg'
        mock_bids_path.fpath = '/tmp/test.fif'
        
        mock_find_paths.return_value = [mock_bids_path]
        
        # Create a mock raw object with proper MNE Info object
        mock_raw = Mock()
        # Create a real MNE Info object for the mock
        info = mne.create_info(
            ch_names=['EEG001', 'EEG002'],
            sfreq=500,
            ch_types='eeg'
        )
        mock_raw.info = info
        mock_raw.n_times = 1000
        mock_raw.get_data = Mock(return_value=np.random.randn(2, 1000))
        mock_read_raw.return_value = mock_raw
        
        # Setup Progress mock
        mock_progress_instance = MagicMock()
        mock_progress_class.return_value.__enter__ = Mock(return_value=mock_progress_instance)
        mock_progress_class.return_value.__exit__ = Mock(return_value=False)
        
        # Create pipeline with minimal config (just load_data to avoid complexity)
        pipeline = EEGPreprocessingPipeline(
            bids_root='/tmp/test',
            config={'pipeline': [{'name': 'load_data'}]}  # Minimal pipeline for testing
        )
        
        # Run pipeline
        results = pipeline.run_pipeline(subjects='01', tasks='rest')
        
        # Verify Progress was created
        assert mock_progress_class.called, "Progress should be instantiated"
        
        # Verify logger was used
        assert mock_logger.info.called, "Logger should be called"
        
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
        test_run_pipeline_creates_progress()
        
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
