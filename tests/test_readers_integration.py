#!/usr/bin/env python3
"""
Integration test for readers with EEGPreprocessingPipeline.

This test verifies that both BIDS and Glob readers work correctly
when integrated with the full pipeline.
"""

import sys
import tempfile
from pathlib import Path

# Find the repository root
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"

# Add src to path
sys.path.insert(0, str(src_dir))


def create_mock_bids_dataset(bids_root):
    """Create a minimal mock BIDS dataset for testing."""
    bids_root = Path(bids_root)
    
    # Create subjects
    subjects = ['01', '02']
    tasks = ['rest']
    sessions = ['01']
    
    for sub in subjects:
        sub_dir = bids_root / f'sub-{sub}'
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        for ses in sessions:
            ses_dir = sub_dir / f'ses-{ses}'
            eeg_dir = ses_dir / 'eeg'
            eeg_dir.mkdir(parents=True, exist_ok=True)
            
            for task in tasks:
                # Create minimal BIDS files
                filename = f'sub-{sub}_ses-{ses}_task-{task}_eeg.vhdr'
                (eeg_dir / filename).touch()
    
    return bids_root


def create_mock_glob_dataset(data_root):
    """Create a minimal dataset for glob pattern testing."""
    data_root = Path(data_root)
    
    # Create a simple structure
    subjects = ['01', '02']
    sessions = ['01']
    tasks = ['rest']
    
    for sub in subjects:
        for ses in sessions:
            for task in tasks:
                file_dir = data_root / 'data' / f'sub-{sub}' / f'ses-{ses}' / 'eeg'
                file_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f'sub-{sub}_ses-{ses}_task-{task}_eeg.vhdr'
                (file_dir / filename).touch()
    
    return data_root


def test_pipeline_with_bids_reader():
    """Test that pipeline works with BIDSReader (default behavior)."""
    from eeg_preprocessing_pipeline import EEGPreprocessingPipeline
    
    with tempfile.TemporaryDirectory() as tmpdir:
        bids_root = create_mock_bids_dataset(tmpdir)
        
        # Test 1: Pipeline without explicit reader (should use BIDS by default)
        pipeline = EEGPreprocessingPipeline(bids_root=bids_root, config={})
        
        # Verify the reader is a BIDSReader
        from readers import BIDSReader
        assert isinstance(pipeline.reader, BIDSReader), \
            f"Expected BIDSReader, got {type(pipeline.reader)}"
        
        print("✓ Pipeline uses BIDSReader by default")
        
        # Test 2: Find recordings
        recordings = pipeline.reader.find_recordings(subjects='01', tasks='rest')
        assert len(recordings) > 0, "Should find recordings"
        assert recordings[0]['metadata']['subject'] == '01', \
            f"Expected subject '01', got {recordings[0]['metadata']['subject']}"
        
        print("✓ BIDSReader finds recordings correctly")
        
    print("✓ Pipeline integrates correctly with BIDSReader")


def test_pipeline_with_glob_reader():
    """Test that pipeline works with GlobReader."""
    from eeg_preprocessing_pipeline import EEGPreprocessingPipeline
    from readers import GlobReader
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = create_mock_glob_dataset(tmpdir)
        
        # Create a glob reader
        pattern = "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_eeg.vhdr"
        reader = GlobReader(data_root, pattern)
        
        # Create pipeline with glob reader
        pipeline = EEGPreprocessingPipeline(
            bids_root=data_root,
            config={},
            reader=reader
        )
        
        # Verify the reader is a GlobReader
        assert isinstance(pipeline.reader, GlobReader), \
            f"Expected GlobReader, got {type(pipeline.reader)}"
        
        print("✓ Pipeline accepts GlobReader")
        
        # Test finding recordings
        recordings = pipeline.reader.find_recordings(subjects='01', tasks='rest')
        assert len(recordings) > 0, "Should find recordings"
        assert recordings[0]['metadata']['subject'] == '01', \
            f"Expected subject '01', got {recordings[0]['metadata']['subject']}"
        
        print("✓ GlobReader finds recordings correctly")
        
    print("✓ Pipeline integrates correctly with GlobReader")


def test_readers_return_compatible_structure():
    """Test that both readers return the same data structure."""
    from readers import BIDSReader, GlobReader
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create BIDS dataset
        bids_root = create_mock_bids_dataset(tmpdir)
        bids_reader = BIDSReader(bids_root)
        bids_recordings = bids_reader.find_recordings(subjects='01', tasks='rest')
        
        # Create glob dataset
        glob_root = Path(tmpdir) / 'glob'
        glob_root.mkdir()
        create_mock_glob_dataset(glob_root)
        
        pattern = "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_eeg.vhdr"
        glob_reader = GlobReader(glob_root, pattern)
        glob_recordings = glob_reader.find_recordings(subjects='01', tasks='rest')
        
        # Both should return recordings
        assert len(bids_recordings) > 0, "BIDS reader should find recordings"
        assert len(glob_recordings) > 0, "Glob reader should find recordings"
        
        # Check structure consistency
        for recordings in [bids_recordings, glob_recordings]:
            recording = recordings[0]
            assert 'paths' in recording, "Recording should have 'paths'"
            assert 'metadata' in recording, "Recording should have 'metadata'"
            assert 'recording_name' in recording, "Recording should have 'recording_name'"
            
            assert isinstance(recording['paths'], list), "'paths' should be a list"
            assert isinstance(recording['metadata'], dict), "'metadata' should be a dict"
            assert isinstance(recording['recording_name'], str), "'recording_name' should be a string"
            
            # Check metadata has the expected fields
            assert 'subject' in recording['metadata'], "Metadata should have 'subject'"
            assert 'task' in recording['metadata'], "Metadata should have 'task'"
        
        print("✓ Both readers return compatible data structures")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Running Reader Integration Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_pipeline_with_bids_reader,
        test_pipeline_with_glob_reader,
        test_readers_return_compatible_structure,
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
        print("SUCCESS: All reader integration tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
