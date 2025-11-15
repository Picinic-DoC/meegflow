# NICE EEG Preprocessing Pipeline

A general EEG preprocessing pipeline using MNE-BIDS for reading data and preprocessing. The pipeline outputs clean preprocessed epochs and comprehensive reports in multiple formats for downstream analysis.

## Features

- **MNE-BIDS Integration**: Seamlessly reads EEG data in BIDS format
- **Comprehensive Preprocessing**: Includes filtering, re-referencing, ICA-based artifact removal, and epoching
- **Multiple Output Formats**:
  - Clean preprocessed epochs in `.fif` format
  - Interactive HTML reports using MNE Report
  - JSON reports for easy downstream processing
- **Configurable**: Fully customizable preprocessing parameters via JSON configuration
- **Command-line Interface**: Easy to use from the terminal or integrate into batch processing scripts

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Laouen/nice-preprocessing.git
cd nice-preprocessing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Command-Line Usage

Run the preprocessing pipeline on a single subject:

```bash
python eeg_preprocessing_pipeline.py \
    --bids-root /path/to/bids/dataset \
    --subject 01 \
    --task rest \
    --session 01
```

### With Custom Configuration

Create a configuration file (see `config_example.json`) and use it:

```bash
python eeg_preprocessing_pipeline.py \
    --bids-root /path/to/bids/dataset \
    --subject 01 \
    --task rest \
    --config my_config.json
```

### Python API Usage

You can also use the pipeline directly in Python:

```python
from eeg_preprocessing_pipeline import EEGPreprocessingPipeline

# Initialize pipeline
pipeline = EEGPreprocessingPipeline(
    bids_root='/path/to/bids/dataset',
    derivatives_root='/path/to/derivatives'
)

# Run preprocessing
results = pipeline.run_pipeline(
    subject='01',
    session='01',
    task='rest',
    apply_ica=True
)

# Access results
epochs = results['epochs']
preprocessing_info = results['preprocessing_info']
```

## Output Structure

The pipeline creates three derivative folders:

```
derivatives/
├── clean_epochs/          # Preprocessed epochs in .fif format
│   └── sub-01_ses-01_task-rest_epo.fif
├── html_reports/          # Interactive HTML reports
│   └── sub-01_ses-01_task-rest_report.html
└── json_reports/          # JSON reports for downstream analysis
    └── sub-01_ses-01_task-rest_report.json
```

### Output Details

1. **clean_epochs/**: Contains MNE epochs objects saved in `.fif` format
   - Can be loaded with `mne.read_epochs()`
   - Includes all preprocessing (filtering, artifact removal, baseline correction)

2. **html_reports/**: Interactive HTML reports generated with MNE Report
   - Raw data visualization with PSD
   - ICA components and artifact detection
   - Cleaned epochs visualization
   - Average evoked responses

3. **json_reports/**: Structured JSON reports containing:
   - Preprocessing parameters used
   - Data quality metrics
   - Number of epochs before/after rejection
   - ICA components excluded
   - Channel information
   - Timestamps and metadata

## Configuration

The pipeline can be configured using a JSON file. Example configuration:

```json
{
  "l_freq": 0.5,              // High-pass filter (Hz)
  "h_freq": 40.0,             // Low-pass filter (Hz)
  "epochs_tmin": -0.2,        // Epoch start time (s)
  "epochs_tmax": 0.8,         // Epoch end time (s)
  "baseline": [null, 0],      // Baseline correction window
  "reject_criteria": {        // Artifact rejection thresholds
    "eeg": 1.5e-04           // 150 µV for EEG
  },
  "ica_n_components": 20,     // Number of ICA components
  "ica_method": "fastica",    // ICA algorithm
  "event_id": null            // Event IDs (null = use all)
}
```

## Command-Line Arguments

- `--bids-root`: Path to BIDS root directory (required)
- `--subject`: Subject ID (required)
- `--session`: Session ID (optional)
- `--task`: Task name (optional)
- `--run`: Run number (optional)
- `--derivatives-root`: Custom derivatives path (optional)
- `--no-ica`: Skip ICA application (optional)
- `--config`: Path to JSON configuration file (optional)

## Preprocessing Steps

The pipeline performs the following steps:

1. **Data Loading**: Reads raw EEG data using MNE-BIDS
2. **Filtering**: Applies bandpass filter (default: 0.5-40 Hz)
3. **Re-referencing**: Sets average reference
4. **ICA**: Removes artifacts using Independent Component Analysis
   - Automatic detection of EOG artifacts
   - Automatic detection of ECG artifacts
5. **Epoching**: Creates epochs around events
6. **Artifact Rejection**: Removes bad epochs based on amplitude criteria
7. **Output Generation**: Saves epochs and generates reports

## Batch Processing

For processing multiple subjects, create a simple bash script:

```bash
#!/bin/bash

BIDS_ROOT="/path/to/bids/dataset"
SUBJECTS=("01" "02" "03" "04")

for subject in "${SUBJECTS[@]}"; do
    python eeg_preprocessing_pipeline.py \
        --bids-root "$BIDS_ROOT" \
        --subject "$subject" \
        --task rest \
        --session 01
done
```

## SLURM Integration

For HPC environments with SLURM, create a submission script:

```bash
#!/bin/bash
#SBATCH --job-name=eeg_preproc
#SBATCH --output=logs/preproc_%A_%a.out
#SBATCH --error=logs/preproc_%A_%a.err
#SBATCH --array=1-20
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Load required modules (adjust for your cluster)
module load python/3.9

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Get subject ID from array task ID
SUBJECT=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

# Run preprocessing
python eeg_preprocessing_pipeline.py \
    --bids-root /path/to/bids/dataset \
    --subject $SUBJECT \
    --task rest \
    --session 01
```

## Requirements

- Python >= 3.8
- mne >= 1.5.0
- mne-bids >= 0.14
- numpy >= 1.24.0
- scipy >= 1.11.0
- matplotlib >= 3.7.0 (recommended)
- pandas >= 2.0.0 (recommended)

## License

This project is ready to use for several projects and includes scripts for SLURM execution.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on the GitHub repository. 
