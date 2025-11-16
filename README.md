# NICE EEG Preprocessing Pipeline

A modular, configuration-driven EEG preprocessing pipeline using MNE-BIDS. The pipeline uses auxiliary functions for each preprocessing step, allowing you to choose which steps to run, their order, and their parameters through a simple YAML configuration.

## Features

- **MNE-BIDS Integration**: Seamlessly reads EEG data in BIDS format
- **Modular Design**: Each preprocessing step is a separate function
- **Configuration-Driven**: Choose steps, their order, and parameters via YAML
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
    --tasks rest \
    --config config_example.yaml
```

Process all subjects with a specific task:

```bash
python eeg_preprocessing_pipeline.py \
    --bids-root /path/to/bids/dataset \
    --tasks rest
```

Process specific subjects with multiple tasks:

```bash
python eeg_preprocessing_pipeline.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 \
    --tasks rest task1 task2
```

### Python API Usage

You can also use the pipeline directly in Python:

```python
from eeg_preprocessing_pipeline import EEGPreprocessingPipeline

# Load configuration
import yaml
with open('config_example.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
pipeline = EEGPreprocessingPipeline(
    bids_root='/path/to/bids/dataset',
    output_root='/path/to/derivatives',
    config=config
)

# Run preprocessing on multiple subjects
results = pipeline.run_pipeline(
    subjects=['01', '02', '03'],
    tasks='rest'
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

The pipeline is configuration-driven. You define a list of preprocessing steps, their order, and parameters in a YAML file.

### Available Steps

- **load_data**: Load raw data into memory
- **filter**: Apply bandpass filtering
- **reference**: Apply re-referencing
- **ica**: ICA-based artifact removal
- **find_events**: Find events in the data
- **epoch**: Create epochs around events
- **find_bads_channels_threshold**: Find bad channels using threshold-based rejection
- **find_bads_channels_variance**: Find bad channels using variance-based detection
- **find_bads_channels_high_frequency**: Find bad channels using high-frequency variance
- **find_bads_epochs_threshold**: Find and remove bad epochs using threshold-based rejection
- **save_clean_epochs**: Save epochs to .fif file
- **save_clean_raw**: Save raw data to .fif file
- **generate_report**: Generate JSON and HTML reports

### Example Configuration

See `config_example.yaml` for a full pipeline with epochs:

```yaml
pipeline:
  - name: load_data
  - name: filter
    l_freq: 0.5
    h_freq: 40.0
  - name: reference
    type: average
  - name: ica
    n_components: 20
    find_eog: true
    apply: true
  - name: find_events
  - name: epoch
    tmin: -0.2
    tmax: 0.8
    baseline: [null, 0]
  - name: save_clean_epochs
  - name: generate_report
    generate_html: true
```

See `config_raw_only.yaml` for a simpler pipeline without epoching:

```yaml
pipeline:
  - name: load_data
  - name: filter
    l_freq: 1.0
    h_freq: 30.0
  - name: reference
    type: average
  - name: ica
    n_components: 15
    find_eog: true
    apply: true
  - name: save_clean_raw
  - name: generate_report
```

See `config_with_adaptive_reject.yaml` for a pipeline with adaptive autoreject steps:

```yaml
pipeline:
  - name: load_data
  - name: filter
    l_freq: 0.5
    h_freq: 40.0
  - name: reference
    type: average
  - name: ica
    n_components: 20
    find_eog: true
    apply: true
  - name: find_events
  - name: epoch
    tmin: -0.2
    tmax: 0.8
    baseline: [null, 0]
  - name: find_bads_channels_threshold
    reject:
      eeg: 1.5e-04
    n_epochs_bad_ch: 0.5
  - name: find_bads_channels_variance
    zscore_thresh: 4
  - name: find_bads_channels_high_frequency
    zscore_thresh: 4
  - name: find_bads_epochs_threshold
    reject:
      eeg: 1.5e-04
    n_channels_bad_epoch: 0.1
  - name: save_clean_epochs
  - name: generate_report
```

## Command-Line Arguments

### Required Arguments
- `--bids-root`: Path to BIDS root directory

### Optional Filter Arguments
These arguments use the same matching logic as `mne-bids` `find_matching_paths`. If not specified, all matching files will be processed.

- `--subjects`: Subject ID(s) to process, space-separated (e.g., `--subjects 01 02 03`)
- `--sessions`: Session ID(s) to process, space-separated
- `--tasks`: Task name(s) to process, space-separated (e.g., `--tasks rest task1`)
- `--acquisitions`: Acquisition parameter(s) to process
- `--runs`: Run number(s) to process
- `--processings`: Processing label(s) to process
- `--recordings`: Recording name(s) to process
- `--spaces`: Coordinate space(s) to process
- `--splits`: Split(s) of continuous recording to process
- `--descriptions`: Description(s) to process

### Other Arguments
- `--output-root`: Custom output path (optional, defaults to `bids-root/derivatives/nice-preprocessing`)
- `--config`: Path to YAML configuration file (optional)

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

### 7. find_bads_channels_threshold
Find bad channels using threshold-based rejection. Marks channels as bad if they exceed rejection thresholds in too many epochs.
- `picks`: Channel indices to check (default: EEG channels)
- `reject`: Rejection thresholds by channel type (e.g., `{"eeg": 150e-6}`)
- `n_epochs_bad_ch`: Fraction or number of epochs a channel must be bad in to be marked as bad (default: 0.5)

### 8. find_bads_channels_variance
Find bad channels using variance-based detection. Identifies channels with abnormally high or low variance.
- `instance`: Which data instance to use - 'raw' or 'epochs' (default: 'epochs')
- `picks`: Channel indices to check (default: EEG channels)
- `zscore_thresh`: Z-score threshold for outlier detection (default: 4)
- `max_iter`: Maximum iterations for iterative outlier removal (default: 2)

### 9. find_bads_channels_high_frequency
Find bad channels using high-frequency variance. Detects channels with excessive high-frequency noise.
- `instance`: Which data instance to use - 'raw' or 'epochs' (default: 'epochs')
- `picks`: Channel indices to check (default: EEG channels)
- `zscore_thresh`: Z-score threshold for outlier detection (default: 4)
- `max_iter`: Maximum iterations for iterative outlier removal (default: 2)

### 10. find_bads_epochs_threshold
Find and remove bad epochs using threshold-based rejection. Drops epochs that have too many bad channels.
- `picks`: Channel indices to check (default: EEG channels)
- `reject`: Rejection thresholds by channel type (e.g., `{"eeg": 150e-6}`)
- `n_channels_bad_epoch`: Fraction or number of channels that must be bad for an epoch to be rejected (default: 0.1)

### 11. save_clean_epochs
Save epochs to .fif file. No parameters needed.

### 12. save_clean_raw
Save raw data to .fif file. No parameters needed.

### 13. generate_report
Generate JSON and optionally HTML reports.
- `generate_html`: Whether to generate HTML report (true/false)

## Batch Processing

The pipeline processes multiple subjects and files sequentially. You can process:

```bash
# Process specific subjects with a specific task
python eeg_preprocessing_pipeline.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 03 04 05 \
    --tasks rest \
    --config config_example.yaml

# Process all subjects in the dataset
python eeg_preprocessing_pipeline.py \
    --bids-root /path/to/bids/dataset \
    --config config_example.yaml

# Process specific sessions for specific subjects
python eeg_preprocessing_pipeline.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 \
    --sessions 01 02 \
    --tasks rest
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
