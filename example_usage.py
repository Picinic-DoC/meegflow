#!/usr/bin/env python3
"""
Example usage of the EEG Preprocessing Pipeline.

This script demonstrates how to use the EEG preprocessing pipeline
in various scenarios.
"""

from pathlib import Path
from eeg_preprocessing_pipeline import EEGPreprocessingPipeline


def example_basic_usage():
    """
    Basic usage example: Process a single subject with default settings.
    """
    print("Example 1: Basic Usage")
    print("-" * 50)
    
    # Initialize pipeline
    pipeline = EEGPreprocessingPipeline(
        bids_root='/path/to/bids/dataset'
    )
    
    # Run preprocessing on a single subject
    results = pipeline.run_pipeline(
        subject='01',
        session='01',
        task='rest'
    )
    
    # Access the results
    epochs = results['epochs']
    print(f"Number of clean epochs: {len(epochs)}")
    print(f"Preprocessing info: {results['preprocessing_info']}")
    print()


def example_custom_config():
    """
    Example with custom configuration.
    """
    print("Example 2: Custom Configuration")
    print("-" * 50)
    
    # Define custom configuration
    custom_config = {
        'l_freq': 1.0,          # Higher high-pass filter
        'h_freq': 30.0,         # Lower low-pass filter
        'epochs_tmin': -0.5,    # Longer pre-stimulus window
        'epochs_tmax': 1.5,     # Longer post-stimulus window
        'baseline': (-0.2, 0),  # Custom baseline
        'reject_criteria': {
            'eeg': 100e-6,      # Stricter rejection (100 µV)
        },
        'ica_n_components': 15,
    }
    
    # Initialize pipeline with custom config
    pipeline = EEGPreprocessingPipeline(
        bids_root='/path/to/bids/dataset',
        config=custom_config
    )
    
    # Run preprocessing
    results = pipeline.run_pipeline(
        subject='02',
        task='rest'
    )
    
    print(f"Used filter: {custom_config['l_freq']}-{custom_config['h_freq']} Hz")
    print()


def example_batch_processing():
    """
    Example of batch processing multiple subjects.
    """
    print("Example 3: Batch Processing")
    print("-" * 50)
    
    # List of subjects to process
    subjects = ['01', '02', '03', '04', '05']
    
    # Initialize pipeline
    pipeline = EEGPreprocessingPipeline(
        bids_root='/path/to/bids/dataset'
    )
    
    # Process each subject
    for subject in subjects:
        try:
            print(f"\nProcessing subject {subject}...")
            results = pipeline.run_pipeline(
                subject=subject,
                task='rest',
                session='01'
            )
            print(f"Subject {subject} completed successfully!")
            
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
            continue
    
    print("\nBatch processing completed!")
    print()


def example_without_ica():
    """
    Example without ICA application.
    """
    print("Example 4: Preprocessing without ICA")
    print("-" * 50)
    
    # Initialize pipeline
    pipeline = EEGPreprocessingPipeline(
        bids_root='/path/to/bids/dataset'
    )
    
    # Run preprocessing without ICA
    results = pipeline.run_pipeline(
        subject='01',
        task='rest',
        apply_ica=False  # Skip ICA
    )
    
    print("Preprocessing completed without ICA")
    print()


def example_custom_derivatives_location():
    """
    Example with custom derivatives location.
    """
    print("Example 5: Custom Derivatives Location")
    print("-" * 50)
    
    # Initialize pipeline with custom derivatives location
    pipeline = EEGPreprocessingPipeline(
        bids_root='/path/to/bids/dataset',
        derivatives_root='/path/to/custom/derivatives'
    )
    
    # Run preprocessing
    results = pipeline.run_pipeline(
        subject='01',
        task='rest'
    )
    
    print(f"Outputs saved to: /path/to/custom/derivatives")
    print()


def example_accessing_outputs():
    """
    Example of accessing and using the outputs.
    """
    print("Example 6: Accessing Outputs")
    print("-" * 50)
    
    # Initialize pipeline
    pipeline = EEGPreprocessingPipeline(
        bids_root='/path/to/bids/dataset'
    )
    
    # Run preprocessing
    results = pipeline.run_pipeline(
        subject='01',
        task='rest'
    )
    
    # Access different outputs
    epochs = results['epochs']
    raw = results['raw']
    ica = results['ica']
    info = results['preprocessing_info']
    
    # Use the epochs object
    print(f"Epochs shape: {epochs.get_data().shape}")
    print(f"Channel names: {epochs.ch_names}")
    print(f"Sampling frequency: {epochs.info['sfreq']} Hz")
    
    # Compute evoked response
    evoked = epochs.average()
    print(f"Evoked data shape: {evoked.data.shape}")
    
    # Access preprocessing information
    print(f"\nPreprocessing Information:")
    print(f"Number of epochs kept: {info['epochs']['n_epochs_after_rejection']}")
    print(f"Rejection rate: {info['epochs']['rejection_rate']:.2f}%")
    print(f"ICA components excluded: {info['ica']['excluded_components']}")
    print()


def example_loading_saved_epochs():
    """
    Example of loading previously saved epochs.
    """
    print("Example 7: Loading Saved Epochs")
    print("-" * 50)
    
    import mne
    
    # Path to saved epochs
    epochs_path = '/path/to/derivatives/clean_epochs/sub-01_task-rest_epo.fif'
    
    # Load epochs
    epochs = mne.read_epochs(epochs_path, preload=True)
    
    print(f"Loaded {len(epochs)} epochs")
    print(f"Channels: {epochs.ch_names}")
    
    # You can now use the epochs for further analysis
    # For example, compute ERPs, time-frequency analysis, etc.
    print()


def example_reading_json_report():
    """
    Example of reading and using the JSON report.
    """
    print("Example 8: Reading JSON Report")
    print("-" * 50)
    
    import json
    
    # Path to JSON report
    json_path = '/path/to/derivatives/json_reports/sub-01_task-rest_report.json'
    
    # Load JSON report
    with open(json_path, 'r') as f:
        report = json.load(f)
    
    # Access different parts of the report
    print(f"Preprocessing date: {report['preprocessing_date']}")
    print(f"Subject: {report['subject']}")
    print(f"Task: {report['task']}")
    print(f"Number of channels: {report['preprocessing_info']['n_channels']}")
    print(f"Sampling rate: {report['preprocessing_info']['sampling_rate']} Hz")
    print(f"Filter applied: {report['preprocessing_info']['filter_applied']}")
    print(f"Epochs kept: {report['preprocessing_info']['epochs']['n_epochs_after_rejection']}")
    print()


if __name__ == '__main__':
    print("=" * 50)
    print("EEG Preprocessing Pipeline - Usage Examples")
    print("=" * 50)
    print()
    
    # Note: These examples use placeholder paths.
    # Replace with actual paths to your BIDS dataset to run them.
    
    print("Note: These are example snippets. Update the paths to your")
    print("actual BIDS dataset to run them.")
    print()
    
    # Uncomment the examples you want to run:
    # example_basic_usage()
    # example_custom_config()
    # example_batch_processing()
    # example_without_ica()
    # example_custom_derivatives_location()
    # example_accessing_outputs()
    # example_loading_saved_epochs()
    # example_reading_json_report()
