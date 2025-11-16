# NICE EEG Preprocessing Pipeline

A modular, configuration-driven EEG preprocessing pipeline using MNE-BIDS. The pipeline uses auxiliary functions for each preprocessing step, allowing you to choose which steps to run, their order, and their parameters through a simple JSON configuration.

## Features

- **MNE-BIDS Integration**: Seamlessly reads EEG data in BIDS format
- **Modular Design**: Each preprocessing step is a separate function
- **Configuration-Driven**: Choose steps, their order, and parameters via JSON
- **Multiple Output Formats**:
  - Clean preprocessed epochs in `.fif` format
  - Clean preprocessed raw data in `.fif` format
  - Interactive HTML reports using MNE Report
  - JSON reports for easy downstream processing
- **Batch Processing**: Process multiple subjects sequentially
- **Command-line Interface**: Easy to use from the terminal

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

### Process Multiple Subjects

Run the preprocessing pipeline on multiple subjects:

```bash
python eeg_preprocessing_pipeline.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 03 \
    --task rest \
    --config config_example.json
```

Or with comma-separated subjects:

```bash
python eeg_preprocessing_pipeline.py \
    --bids-root /path/to/bids/dataset \
    --subjects "01,02,03" \
    --task rest
```

### Python API Usage

You can also use the pipeline directly in Python:

```python
from eeg_preprocessing_pipeline import EEGPreprocessingPipeline

# Load configuration
import json
with open('config_example.json', 'r') as f:
    config = json.load(f)

# Initialize pipeline
pipeline = EEGPreprocessingPipeline(
    bids_root='/path/to/bids/dataset',
    output_root='/path/to/derivatives',
    config=config
)

# Run preprocessing on multiple subjects
results = pipeline.run_pipeline(
    subjects=['01', '02', '03'],
    task='rest'
)

# Access results for each subject
for subject, result in results.items():
    print(f"Subject {subject}: {result}")
```

## Output Structure

The pipeline creates subject-specific folders with outputs based on the configured steps:

```
derivatives/nice-preprocessing/
└── sub-01/
    ├── clean_epochs/
    │   └── sub-01_task-rest_clean-epo.fif
    ├── clean_raw/
    │   └── sub-01_raw_clean.fif
    └── reports/
        ├── preprocessing_report.json
        └── preprocessing_report.html
```

### Output Details

1. **clean_epochs/**: Contains MNE epochs objects saved in `.fif` format (if epoching step is included)
   - Can be loaded with `mne.read_epochs()`
   - Includes all preprocessing (filtering, artifact removal, baseline correction)

2. **clean_raw/**: Contains preprocessed raw data in `.fif` format (if save_clean_raw step is included)
   - Can be loaded with `mne.io.read_raw_fif()`
   - Includes preprocessing up to that point in the pipeline

3. **reports/**: Contains preprocessing reports
   - **JSON report**: Preprocessing parameters, quality metrics, steps performed
   - **HTML report**: Interactive visualization (optional, if generate_html is true)

## Configuration

The pipeline is configuration-driven. You define a list of preprocessing steps, their order, and parameters in a JSON file.

### Available Steps

- **load_data**: Load raw data into memory
- **filter**: Apply bandpass filtering
- **reference**: Apply re-referencing
- **ica**: ICA-based artifact removal
- **find_events**: Find events in the data
- **epoch**: Create epochs around events
- **save_clean_epochs**: Save epochs to .fif file
- **save_clean_raw**: Save raw data to .fif file
- **generate_report**: Generate JSON and HTML reports

### Example Configuration

See `config_example.json` for a full pipeline with epochs:

```json
{
  "pipeline": [
    {"name": "load_data"},
    {"name": "filter", "l_freq": 0.5, "h_freq": 40.0},
    {"name": "reference", "type": "average"},
    {"name": "ica", "n_components": 20, "find_eog": true, "apply": true},
    {"name": "find_events"},
    {"name": "epoch", "tmin": -0.2, "tmax": 0.8, "baseline": [null, 0]},
    {"name": "save_clean_epochs"},
    {"name": "generate_report", "generate_html": true}
  ]
}
```

See `config_raw_only.json` for a simpler pipeline without epoching:

```json
{
  "pipeline": [
    {"name": "load_data"},
    {"name": "filter", "l_freq": 1.0, "h_freq": 30.0},
    {"name": "reference", "type": "average"},
    {"name": "ica", "n_components": 15, "find_eog": true, "apply": true},
    {"name": "save_clean_raw"},
    {"name": "generate_report"}
  ]
}
```

## Command-Line Arguments

- `--bids-root`: Path to BIDS root directory (required)
- `--subjects`: Subject ID(s) to process, space or comma-separated (required)
- `--task`: Task name (optional)
- `--output-root`: Custom output path (optional, defaults to `bids-root/derivatives/nice-preprocessing`)
- `--config`: Path to JSON configuration file (optional)

## Preprocessing Steps Details

Each step can be customized through the configuration:

### 1. load_data
Loads raw data into memory. No parameters needed.

### 2. filter
Apply bandpass filtering.
- `l_freq`: High-pass filter frequency (Hz)
- `h_freq`: Low-pass filter frequency (Hz)

### 3. reference
Apply re-referencing.
- `type`: Reference type ('average' or other)
- `projection`: Whether to use projection (true/false)

### 4. ica
ICA-based artifact removal.
- `n_components`: Number of ICA components
- `method`: ICA method ('fastica', 'infomax', 'picard')
- `find_eog`: Automatically find EOG artifacts (true/false)
- `find_ecg`: Automatically find ECG artifacts (true/false)
- `apply`: Apply ICA to remove artifacts (true/false)

### 5. find_events
Find events in the data.
- `shortest_event`: Minimum event duration in samples

### 6. epoch
Create epochs around events.
- `tmin`: Start time before event (seconds)
- `tmax`: End time after event (seconds)
- `baseline`: Baseline correction window (tuple or null)
- `event_id`: Event IDs to include (dict or null for all)
- `reject`: Rejection criteria (dict with channel type keys)

### 7. save_clean_epochs
Save epochs to .fif file. No parameters needed.

### 8. save_clean_raw
Save raw data to .fif file. No parameters needed.

### 9. generate_report
Generate JSON and optionally HTML reports.
- `generate_html`: Whether to generate HTML report (true/false)

## Batch Processing

The pipeline now processes multiple subjects sequentially. Simply pass multiple subject IDs:

```bash
# Process multiple subjects at once
python eeg_preprocessing_pipeline.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 03 04 05 \
    --task rest \
    --config config_example.json
```

For HPC/cluster environments, you can create your own SLURM or other batch submission scripts that call the pipeline with subject lists.

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
