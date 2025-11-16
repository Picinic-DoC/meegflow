#!/usr/bin/env python3
"""
Functional tests for adaptive_reject functions with MNE >= 1.5.0

These tests verify that the adaptive_reject functions work correctly with
public MNE APIs and do not use deprecated private methods.
"""

import sys
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    import mne
    from mne.utils import logger
    import adaptive_reject
    MNE_AVAILABLE = True
except ImportError as e:
    MNE_AVAILABLE = False
    print(f"Warning: Could not import required modules: {e}")


def create_test_epochs():
    """Create simple test epochs for functional testing."""
    # Create simple raw data
    n_channels = 5
    n_times = 1000
    sfreq = 100.0
    
    # Create some data with one "bad" channel (very high amplitude)
    data = np.random.randn(n_channels, n_times) * 1e-6
    data[2, :] = np.random.randn(n_times) * 5e-5  # Bad channel with high amplitude
    
    info = mne.create_info(
        ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types='eeg'
    )
    
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Create events and epochs
    n_epochs = 10
    events = np.column_stack([
        np.arange(0, n_epochs * 200, 200),  # Event times
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int)
    ])
    
    epochs = mne.Epochs(
        raw, events, event_id={'test': 1},
        tmin=0.0, tmax=0.5, baseline=None,
        preload=True, verbose=False
    )
    
    return epochs, raw


def test_find_bads_channels_threshold():
    """Test find_bads_channels_threshold with public MNE APIs."""
    if not MNE_AVAILABLE:
        print("✗ Skipped: MNE not available")
        return False
    
    print("Testing find_bads_channels_threshold...")
    
    try:
        epochs, _ = create_test_epochs()
        picks = mne.pick_types(epochs.info, eeg=True)
        reject = {'eeg': 1e-4}
        
        bad_chs = adaptive_reject.find_bads_channels_threshold(
            epochs, picks, reject, n_epochs_bad_ch=0.5
        )
        
        print(f"  Found bad channels: {bad_chs}")
        print("  ✓ find_bads_channels_threshold works with public APIs")
        return True
        
    except AttributeError as e:
        if "_find_outliers" in str(e) or "_data" in str(e):
            print(f"  ✗ Still using private MNE APIs: {e}")
            return False
        raise
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_find_bads_channels_variance():
    """Test find_bads_channels_variance with public MNE APIs."""
    if not MNE_AVAILABLE:
        print("✗ Skipped: MNE not available")
        return False
    
    print("Testing find_bads_channels_variance...")
    
    try:
        epochs, _ = create_test_epochs()
        picks = mne.pick_types(epochs.info, eeg=True)
        
        bad_chs = adaptive_reject.find_bads_channels_variance(
            epochs, picks, zscore_thresh=4, max_iter=2
        )
        
        print(f"  Found bad channels: {bad_chs}")
        print("  ✓ find_bads_channels_variance works with public APIs")
        return True
        
    except AttributeError as e:
        if "_find_outliers" in str(e) or "_data" in str(e):
            print(f"  ✗ Still using private MNE APIs: {e}")
            return False
        raise
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_find_bads_channels_high_frequency():
    """Test find_bads_channels_high_frequency with public MNE APIs."""
    if not MNE_AVAILABLE:
        print("✗ Skipped: MNE not available")
        return False
    
    print("Testing find_bads_channels_high_frequency...")
    
    try:
        epochs, _ = create_test_epochs()
        picks = mne.pick_types(epochs.info, eeg=True)
        
        bad_chs = adaptive_reject.find_bads_channels_high_frequency(
            epochs, picks, zscore_thresh=4, max_iter=2
        )
        
        print(f"  Found bad channels: {bad_chs}")
        print("  ✓ find_bads_channels_high_frequency works with public APIs")
        return True
        
    except AttributeError as e:
        if "_find_outliers" in str(e) or "_data" in str(e):
            print(f"  ✗ Still using private MNE APIs: {e}")
            return False
        raise
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_find_bads_epochs_threshold():
    """Test find_bads_epochs_threshold with public MNE APIs."""
    if not MNE_AVAILABLE:
        print("✗ Skipped: MNE not available")
        return False
    
    print("Testing find_bads_epochs_threshold...")
    
    try:
        epochs, _ = create_test_epochs()
        picks = mne.pick_types(epochs.info, eeg=True)
        reject = {'eeg': 1e-4}
        
        bad_epochs = adaptive_reject.find_bads_epochs_threshold(
            epochs, picks, reject, n_channels_bad_epoch=0.1
        )
        
        print(f"  Found bad epochs: {bad_epochs}")
        print("  ✓ find_bads_epochs_threshold works with public APIs")
        return True
        
    except AttributeError as e:
        if "_find_outliers" in str(e) or "_data" in str(e):
            print(f"  ✗ Still using private MNE APIs: {e}")
            return False
        raise
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_raw_data():
    """Test variance and high frequency functions with raw data."""
    if not MNE_AVAILABLE:
        print("✗ Skipped: MNE not available")
        return False
    
    print("Testing with raw data (not epochs)...")
    
    try:
        _, raw = create_test_epochs()
        picks = mne.pick_types(raw.info, eeg=True)
        
        # Test variance
        bad_chs_var = adaptive_reject.find_bads_channels_variance(
            raw, picks, zscore_thresh=4, max_iter=2
        )
        print(f"  Raw variance: Found bad channels: {bad_chs_var}")
        
        # Test high frequency
        bad_chs_hf = adaptive_reject.find_bads_channels_high_frequency(
            raw, picks, zscore_thresh=4, max_iter=2
        )
        print(f"  Raw high freq: Found bad channels: {bad_chs_hf}")
        
        print("  ✓ Functions work with raw data using public APIs")
        return True
        
    except AttributeError as e:
        if "_find_outliers" in str(e) or "_data" in str(e):
            print(f"  ✗ Still using private MNE APIs: {e}")
            return False
        raise
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_private_imports():
    """Verify that no private MNE methods are imported."""
    print("Checking for private API usage in code...")
    
    try:
        import inspect
        source = inspect.getsource(adaptive_reject)
        
        # Check for private method usage
        if 'mne.preprocessing.bads._find_outliers' in source:
            print("  ✗ Still importing mne.preprocessing.bads._find_outliers")
            return False
        
        if '._data' in source:
            print("  ✗ Still using ._data attribute")
            return False
        
        print("  ✓ No private MNE API usage detected in source code")
        return True
        
    except Exception as e:
        print(f"  ✗ Error checking source: {e}")
        return False


def run_all_tests():
    """Run all functional tests."""
    print("=" * 60)
    print("Running Adaptive Reject Functional Tests (MNE >= 1.5.0)")
    print("=" * 60)
    print()
    
    if not MNE_AVAILABLE:
        print("MNE is not available. Please install requirements.")
        return 1
    
    print(f"MNE version: {mne.__version__}")
    print()
    
    tests = [
        test_no_private_imports,
        test_find_bads_channels_threshold,
        test_find_bads_channels_variance,
        test_find_bads_channels_high_frequency,
        test_find_bads_epochs_threshold,
        test_with_raw_data,
    ]
    
    failed_tests = []
    
    for test in tests:
        try:
            if not test():
                failed_tests.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} raised exception: {e}")
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
        print("SUCCESS: All functional tests passed!")
        print("The adaptive_reject functions are compliant with MNE >= 1.5.0")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
