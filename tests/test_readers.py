#!/usr/bin/env python3
"""
Tests for readers module.

This test verifies that:
1. BIDSReader works correctly and maintains backward compatibility
2. GlobReader correctly extracts variables from patterns
3. Both readers return consistent data structures
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
    subjects = ['01', '02', '03']
    tasks = ['rest', 'task1']
    sessions = ['01', '02']
    
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
    subjects = ['01', '02', '03']
    sessions = ['01', '02']
    tasks = ['rest', 'task1']
    
    for sub in subjects:
        for ses in sessions:
            for task in tasks:
                file_dir = data_root / 'data' / f'sub-{sub}' / f'ses-{ses}' / 'eeg'
                file_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f'sub-{sub}_ses-{ses}_task-{task}_eeg.vhdr'
                (file_dir / filename).touch()
    
    return data_root


def test_bids_reader_basic():
    """Test BIDSReader basic functionality."""
    from readers import BIDSReader
    
    with tempfile.TemporaryDirectory() as tmpdir:
        bids_root = create_mock_bids_dataset(tmpdir)
        reader = BIDSReader(bids_root)
        
        # Test finding all recordings
        recordings = reader.find_recordings()
        
        assert len(recordings) > 0, f"Expected recordings, got {len(recordings)}"
        
        # Check structure
        for recording in recordings:
            assert 'paths' in recording, "Recording should have 'paths'"
            assert 'metadata' in recording, "Recording should have 'metadata'"
            assert 'recording_name' in recording, "Recording should have 'recording_name'"
            
            metadata = recording['metadata']
            assert 'subject' in metadata, "Metadata should have 'subject'"
            assert 'task' in metadata, "Metadata should have 'task'"
            
    print("✓ BIDSReader basic functionality works")


def test_bids_reader_filtering():
    """Test BIDSReader filtering by subject and task."""
    from readers import BIDSReader
    
    with tempfile.TemporaryDirectory() as tmpdir:
        bids_root = create_mock_bids_dataset(tmpdir)
        reader = BIDSReader(bids_root)
        
        # Test filtering by subject
        recordings = reader.find_recordings(subjects='01')
        
        for recording in recordings:
            assert recording['metadata']['subject'] == '01', \
                f"Expected subject '01', got {recording['metadata']['subject']}"
        
        # Test filtering by task
        recordings = reader.find_recordings(tasks='rest')
        
        for recording in recordings:
            assert recording['metadata']['task'] == 'rest', \
                f"Expected task 'rest', got {recording['metadata']['task']}"
            
    print("✓ BIDSReader filtering works correctly")


def test_glob_reader_variable_extraction():
    """Test GlobReader extracts variables correctly."""
    from readers import GlobReader
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = create_mock_glob_dataset(tmpdir)
        
        pattern = "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_eeg.vhdr"
        reader = GlobReader(data_root, pattern)
        
        # Check pattern parsing
        assert 'subject' in reader.variable_names, "Should extract 'subject' variable"
        assert 'session' in reader.variable_names, "Should extract 'session' variable"
        assert 'task' in reader.variable_names, "Should extract 'task' variable"
        
    print("✓ GlobReader variable extraction works")


def test_glob_reader_find_recordings():
    """Test GlobReader finds recordings correctly."""
    from readers import GlobReader
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = create_mock_glob_dataset(tmpdir)
        
        pattern = "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_eeg.vhdr"
        reader = GlobReader(data_root, pattern)
        
        # Find all recordings
        recordings = reader.find_recordings()
        
        assert len(recordings) > 0, f"Expected recordings, got {len(recordings)}"
        
        # Check structure matches BIDSReader
        for recording in recordings:
            assert 'paths' in recording, "Recording should have 'paths'"
            assert 'metadata' in recording, "Recording should have 'metadata'"
            assert 'recording_name' in recording, "Recording should have 'recording_name'"
            
            metadata = recording['metadata']
            assert 'subject' in metadata, "Metadata should have 'subject'"
            assert 'task' in metadata, "Metadata should have 'task'"
            assert 'session' in metadata, "Metadata should have 'session'"
            
    print("✓ GlobReader finds recordings correctly")


def test_glob_reader_filtering():
    """Test GlobReader filtering by criteria."""
    from readers import GlobReader
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = create_mock_glob_dataset(tmpdir)
        
        pattern = "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_eeg.vhdr"
        reader = GlobReader(data_root, pattern)
        
        # Test filtering by subject
        recordings = reader.find_recordings(subjects='01')
        
        assert len(recordings) > 0, "Should find recordings for subject 01"
        
        for recording in recordings:
            assert recording['metadata']['subject'] == '01', \
                f"Expected subject '01', got {recording['metadata']['subject']}"
        
        # Test filtering by task
        recordings = reader.find_recordings(tasks='rest')
        
        assert len(recordings) > 0, "Should find recordings for task rest"
        
        for recording in recordings:
            assert recording['metadata']['task'] == 'rest', \
                f"Expected task 'rest', got {recording['metadata']['task']}"
            
    print("✓ GlobReader filtering works correctly")


def test_glob_reader_regex_pattern():
    """Test GlobReader creates correct regex pattern."""
    from readers import GlobReader
    
    pattern = "data/sub-{subject}/task-{task}.vhdr"
    reader = GlobReader("/tmp", pattern)
    
    # Test the regex matches expected patterns
    test_path = "data/sub-01/task-rest.vhdr"
    match = reader.regex_pattern.match(test_path)
    
    assert match is not None, f"Regex should match {test_path}"
    assert match.group('subject') == '01', "Should extract subject='01'"
    assert match.group('task') == 'rest', "Should extract task='rest'"
    
    print("✓ GlobReader regex pattern works correctly")


def test_readers_consistent_interface():
    """Test that BIDSReader and GlobReader return consistent data structures."""
    from readers import BIDSReader, GlobReader
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test BIDSReader
        bids_root = create_mock_bids_dataset(tmpdir)
        bids_reader = BIDSReader(bids_root)
        bids_recordings = bids_reader.find_recordings(subjects='01', tasks='rest')
        
        # Test GlobReader with a different dataset
        glob_root = Path(tmpdir) / 'glob_test'
        glob_root.mkdir()
        create_mock_glob_dataset(glob_root)
        
        pattern = "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_eeg.vhdr"
        glob_reader = GlobReader(glob_root, pattern)
        glob_recordings = glob_reader.find_recordings(subjects='01', tasks='rest')
        
        # Both should return non-empty lists
        assert len(bids_recordings) > 0, "BIDSReader should find recordings"
        assert len(glob_recordings) > 0, "GlobReader should find recordings"
        
        # Check both have the same structure
        for recording in bids_recordings + glob_recordings:
            assert isinstance(recording, dict), "Recording should be a dict"
            assert 'paths' in recording
            assert 'metadata' in recording
            assert 'recording_name' in recording
            assert isinstance(recording['paths'], list)
            assert isinstance(recording['metadata'], dict)
            assert isinstance(recording['recording_name'], str)
    
    print("✓ Readers have consistent interface")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Readers Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_bids_reader_basic,
        test_bids_reader_filtering,
        test_glob_reader_variable_extraction,
        test_glob_reader_find_recordings,
        test_glob_reader_filtering,
        test_glob_reader_regex_pattern,
        test_readers_consistent_interface,
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
        print("SUCCESS: All readers tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
