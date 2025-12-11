#!/usr/bin/env python3
"""
Test for event filtering functionality in find_events step.

This test verifies that:
1. Events can be filtered to keep only specific event IDs
2. The filtering is correctly applied before epoching
3. The event counts are properly tracked in preprocessing steps
"""

import sys
from pathlib import Path
import numpy as np
import mne

# Find the repository root
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"

# Add src to path
sys.path.insert(0, str(src_dir))


def test_event_filtering_basic():
    """Test basic event filtering functionality."""
    from eeg_preprocessing_pipeline import EEGPreprocessingPipeline
    
    # Create synthetic raw data with events
    info = mne.create_info(
        ch_names=['EEG001', 'EEG002', 'EEG003'],
        sfreq=500,
        ch_types='eeg'
    )
    
    # Create 10 seconds of data
    n_samples = 5000
    data = np.random.randn(3, n_samples) * 1e-6
    raw = mne.io.RawArray(data, info)
    
    # Create annotations with different event IDs
    # We'll create events at different times with IDs: 91, 93, 101, 102, 103
    onset = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    duration = [0.0] * 10
    description = ['91', '93', '101', '102', '91', '93', '101', '103', '91', '93']
    
    annotations = mne.Annotations(onset, duration, description)
    raw.set_annotations(annotations)
    
    # Create pipeline data structure
    pipeline = EEGPreprocessingPipeline(
        bids_root='/tmp/test',
        config={'pipeline': []}
    )
    
    data_dict = {
        'raw': raw,
        'preprocessing_steps': []
    }
    
    # Test find_events step WITHOUT filtering
    step_config_no_filter = {
        'get_events_from': 'annotations',
        'event_id': 'auto'
    }
    
    data_dict = pipeline._step_find_events(data_dict, step_config_no_filter)
    
    # Check that all events were found
    assert len(data_dict['events']) == 10, f"Expected 10 events, got {len(data_dict['events'])}"
    
    # Test find_events step WITH filtering (keep only 91, 93, 101)
    data_dict_filtered = {
        'raw': raw,
        'preprocessing_steps': []
    }
    
    step_config_filter = {
        'get_events_from': 'annotations',
        'event_id': 'auto',
        'keep_event_ids': [91, 93, 101]
    }
    
    data_dict_filtered = pipeline._step_find_events(data_dict_filtered, step_config_filter)
    
    # Check that only events 91, 93, 101 were kept
    # Count: 91 appears 3 times, 93 appears 3 times, 101 appears 2 times = 8 total
    assert len(data_dict_filtered['events']) == 8, \
        f"Expected 8 filtered events (3x91 + 3x93 + 2x101), got {len(data_dict_filtered['events'])}"
    
    # Verify that the filtering worked by checking the event_id_mapping
    # The actual event codes in the events array are mapped from the annotation descriptions
    event_id_mapping = data_dict_filtered['event_id_mapping']
    
    # Get the event codes that correspond to our target annotations (91, 93, 101)
    target_annotations = ['91', '93', '101']
    expected_codes = []
    for ann in target_annotations:
        for desc, code in event_id_mapping.items():
            if str(desc) == ann:
                expected_codes.append(code)
                break
    
    # Verify that only the expected event codes are present in the filtered events
    event_codes_in_filtered = np.unique(data_dict_filtered['events'][:, 2])
    assert set(event_codes_in_filtered) == set(expected_codes), \
        f"Filtered events contain unexpected codes. Got {event_codes_in_filtered}, expected {expected_codes}"
    
    # Check preprocessing steps metadata
    step_info = data_dict_filtered['preprocessing_steps'][-1]
    assert step_info['step'] == 'find_events'
    assert step_info['n_events'] == 8
    assert step_info['n_events_before_filter'] == 10
    assert step_info['keep_event_ids'] == [91, 93, 101]
    
    print("✓ Event filtering test passed")
    return True


def test_event_filtering_with_epoch():
    """Test that filtered events work correctly with epoching."""
    from eeg_preprocessing_pipeline import EEGPreprocessingPipeline
    
    # Create synthetic raw data with events
    info = mne.create_info(
        ch_names=['EEG001', 'EEG002'],
        sfreq=500,
        ch_types='eeg'
    )
    
    # Create 10 seconds of data
    n_samples = 5000
    data = np.random.randn(2, n_samples) * 1e-6
    raw = mne.io.RawArray(data, info)
    
    # Create annotations with different event IDs
    onset = [1.0, 2.0, 3.0, 4.0, 5.0]
    duration = [0.0] * 5
    description = ['91', '93', '101', '102', '103']  # Last two should be filtered out
    
    annotations = mne.Annotations(onset, duration, description)
    raw.set_annotations(annotations)
    
    # Create pipeline
    pipeline = EEGPreprocessingPipeline(
        bids_root='/tmp/test',
        config={'pipeline': []}
    )
    
    data_dict = {
        'raw': raw,
        'preprocessing_steps': []
    }
    
    # Apply find_events with filtering
    step_config_events = {
        'get_events_from': 'annotations',
        'event_id': 'auto',
        'keep_event_ids': [91, 93, 101]
    }
    
    data_dict = pipeline._step_find_events(data_dict, step_config_events)
    
    # Apply epoching
    step_config_epoch = {
        'tmin': -0.2,
        'tmax': 0.5,
        'baseline': (None, 0),
        'event_id': None
    }
    
    data_dict = pipeline._step_epoch(data_dict, step_config_epoch)
    
    # Should have 3 epochs (91, 93, 101)
    assert len(data_dict['epochs']) == 3, \
        f"Expected 3 epochs after filtering, got {len(data_dict['epochs'])}"
    
    print("✓ Event filtering with epoching test passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Event Filtering Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_event_filtering_basic,
        test_event_filtering_with_epoch,
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
        print("SUCCESS: All event filtering tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
