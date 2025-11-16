# Changelog

All notable changes to the NICE EEG Preprocessing Pipeline will be documented in this file.

## [1.0.0] - 2024-11-15

### Added
- Initial release of the EEG preprocessing pipeline
- MNE-BIDS integration for reading EEG data
- Comprehensive preprocessing pipeline including:
  - Bandpass filtering
  - Average re-referencing
  - ICA-based artifact removal
  - Epoching and artifact rejection
- Three output formats:
  - Clean epochs in .fif format (derivatives/clean_epochs/)
  - Interactive HTML reports (derivatives/html_reports/)
  - JSON reports for downstream analysis (derivatives/json_reports/)
- Command-line interface with argparse
- Python API for programmatic use
- Example configuration file (config_example.json)
- Example usage scripts (example_usage.py)
- Batch processing support:
  - Local batch script (run_batch.sh)
  - SLURM cluster script (run_slurm.sh)
- Comprehensive documentation:
  - README.md with installation and usage instructions
  - INTEGRATION_GUIDE.md for project integration
  - Structure tests (test_pipeline_structure.py)

### Features
- Automatic EOG and ECG artifact detection with ICA
- Configurable preprocessing parameters via JSON
- BIDS-compliant output structure
- Detailed preprocessing reports with quality metrics
- Support for multiple sessions, tasks, and runs
- Memory-efficient processing
- Error handling and informative logging

### Dependencies
- mne >= 1.5.0
- mne-bids >= 0.14
- numpy >= 1.24.0
- scipy >= 1.11.0
- matplotlib >= 3.7.0 (recommended)
- pandas >= 2.0.0 (recommended)
