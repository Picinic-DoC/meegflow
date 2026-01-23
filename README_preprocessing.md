# NICE EEG Preprocessing Module

The preprocessing module handles all EEG data preprocessing steps including filtering, artifact removal, bad channel detection, epoching, and more. It saves intermediate results that can be used later for report generation.

## Features

- **Complete EEG Preprocessing**: All preprocessing steps from the original pipeline
- **Intermediate Results**: Saves preprocessed data and metadata for later use
- **Batch Processing**: Process multiple subjects/sessions/tasks
- **Progress Tracking**: Real-time progress bars
- **Flexible Configuration**: YAML-based configuration for all steps

## Installation

### Option 1: Docker (Recommended)

Build the preprocessing Docker image:
```bash
docker build -f Dockerfile.preprocessing -t nice-preprocessing .
```

Or use docker-compose:
```bash
docker-compose build preprocessing
```

### Option 2: Local Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Using Docker

Basic preprocessing with intermediate results saved:
```bash
docker run --rm \
    -v /path/to/bids:/data \
    -v /path/to/configs:/configs \
    nice-preprocessing \
    --bids-root /data \
    --config /configs/config_with_save.yaml
```

Using docker-compose:
```bash
# Set environment variables
export BIDS_ROOT=/path/to/bids
export CONFIG_DIR=/path/to/configs

# Run preprocessing
docker-compose run --rm preprocessing \
    --bids-root /data \
    --config /configs/config_with_save.yaml
```

### Using Command Line

After installation, use the `eeg-preprocess-only` command:

```bash
# Basic usage
eeg-preprocess-only --bids-root /path/to/bids --config config.yaml

# Process specific subjects and tasks
eeg-preprocess-only --bids-root /path/to/bids \
    --subjects 01 02 03 \
    --tasks rest \
    --config config.yaml

# With custom output location
eeg-preprocess-only --bids-root /path/to/bids \
    --output-root /path/to/output \
    --config config.yaml

# With logging
eeg-preprocess-only --bids-root /path/to/bids \
    --config config.yaml \
    --log-file preprocessing.log \
    --log-level DEBUG
```

### Using Python API

```python
from preprocessing_pipeline import PreprocessingPipeline
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
pipeline = PreprocessingPipeline(
    bids_root='/path/to/bids',
    config=config
)

# Run preprocessing
results = pipeline.run_pipeline(
    subjects=['01', '02'],
    tasks='rest'
)
```

## Configuration

### Saving Intermediate Results

To save intermediate results, add the `save_intermediate_results` step to your configuration:

```yaml
pipeline:
  - name: bandpass_filter
    l_freq: 0.5
    h_freq: 40.0
  
  - name: reference
    ref_channels: average
    instance: 'raw'
  
  - name: find_events
    shortest_event: 1
  
  - name: epoch
    tmin: -0.2
    tmax: 0.8
    baseline: [null, 0]
  
  # Save intermediate results (required for report generation)
  - name: save_intermediate_results
    save_raw: true
    save_epochs: true
    save_ica: true
```

### Parameters for save_intermediate_results

- `save_raw` (bool, default: True): Save raw data object
- `save_epochs` (bool, default: True): Save epochs object
- `save_ica` (bool, default: True): Save ICA object if it exists
- `save_events` (bool, default: True): Save events array if it exists

## Output Structure

Intermediate results are saved to:
```
bids_root/derivatives/nice_preprocessing/intermediate/
└── sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}/
    ├── metadata.json          # Preprocessing steps and parameters
    ├── raw.pkl                # Raw data object (if save_raw=true)
    ├── epochs.pkl             # Epochs object (if save_epochs=true)
    ├── ica.pkl                # ICA object (if save_ica=true and ICA was run)
    └── events.pkl             # Events array (if save_events=true)
```

## Available Preprocessing Steps

### Data I/O and Setup
- `set_montage`: Set electrode positions
- `drop_unused_channels`: Remove specific channels
- `copy_instance`: Duplicate a data instance

### Filtering
- `bandpass_filter`: Apply high-pass and low-pass filters
- `notch_filter`: Remove line noise
- `resample`: Change sampling frequency

### Reference
- `reference`: Apply re-referencing (average, specific channels, etc.)

### Bad Channel Detection
- `find_flat_channels`: Detect flat/disconnected channels
- `find_bads_channels_threshold`: Threshold-based bad channel detection
- `find_bads_channels_variance`: Variance-based detection
- `find_bads_channels_high_frequency`: High-frequency noise detection

### Bad Channel Handling
- `interpolate_bad_channels`: Repair bad channels via interpolation
- `drop_bad_channels`: Remove bad channels permanently

### ICA
- `ica`: ICA-based artifact removal with automatic EOG/ECG detection

### Epoching
- `find_events`: Extract events from data
- `epoch`: Create epochs around events
- `chunk_in_epoch`: Create fixed-length epochs
- `find_bads_epochs_threshold`: Detect and remove bad epochs

### Output
- `save_clean_instance`: Save preprocessed raw/epochs to .fif format
- `save_intermediate_results`: Save intermediate data for report generation

## Command-Line Arguments

```
usage: eeg-preprocess-only [-h] --bids-root BIDS_ROOT [--output-root OUTPUT_ROOT]
                           [--subjects SUBJECTS [SUBJECTS ...]]
                           [--sessions SESSIONS [SESSIONS ...]]
                           [--tasks TASKS [TASKS ...]]
                           [--acquisitions ACQUISITIONS [ACQUISITIONS ...]]
                           [--runs RUNS [RUNS ...]]
                           [--extension EXTENSION]
                           [--config CONFIG]
                           [--log-file LOG_FILE]
                           [--log-level {DEBUG,INFO,WARNING,ERROR}]

Required arguments:
  --bids-root BIDS_ROOT     Path to BIDS root directory

Optional filters:
  --subjects                Subject ID(s) to process
  --sessions                Session ID(s) to process
  --tasks                   Task name(s) to process
  --acquisitions            Acquisition parameter(s) to process
  --runs                    Run number(s) to process
  --extension               File extension (default: .vhdr)

Other options:
  --output-root             Custom output path
  --config                  Path to YAML configuration file
  --log-file                Path to log file
  --log-level               Logging level (DEBUG, INFO, WARNING, ERROR)
```

## Next Steps

After preprocessing, use the **Report Generation Module** to create JSON and HTML reports:

See [README_report.md](README_report.md) for details on report generation.

## Examples

### Example 1: Basic Preprocessing

```bash
eeg-preprocess-only \
    --bids-root /data/my_study \
    --config configs/config_with_save.yaml
```

### Example 2: Specific Subjects with Logging

```bash
eeg-preprocess-only \
    --bids-root /data/my_study \
    --subjects 01 02 03 \
    --tasks rest eyesclosed \
    --config configs/config_with_save.yaml \
    --log-file preprocessing.log \
    --log-level INFO
```

### Example 3: Using Docker

```bash
docker run --rm \
    -v /data/my_study:/data \
    -v $(pwd)/configs:/configs \
    nice-preprocessing \
    --bids-root /data \
    --config /configs/config_with_save.yaml \
    --subjects 01 02
```

## Troubleshooting

### Issue: No intermediate results are saved

**Solution**: Make sure you include the `save_intermediate_results` step in your configuration file.

### Issue: Out of memory errors

**Solution**: Process fewer subjects at once or use a machine with more RAM. Consider processing subjects one at a time.

### Issue: Missing channels in output

**Solution**: Check your `drop_unused_channels` and `drop_bad_channels` steps. Use `interpolate_bad_channels` instead of dropping if you want to keep the same channel count.

## See Also

- [Main README](README.md) - Complete pipeline documentation
- [Report Generation README](README_report.md) - Report generation module
- [Configuration Examples](configs/) - Example configuration files
