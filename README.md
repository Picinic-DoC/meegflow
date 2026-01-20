# NICE EEG Preprocessing Pipeline

A modular, configuration-driven EEG preprocessing pipeline using MNE-BIDS. The pipeline uses auxiliary functions for each preprocessing step, allowing you to choose which steps to run, their order, and their parameters through a simple YAML configuration.

## Features

- **MNE-BIDS Integration**: Seamlessly reads EEG data in BIDS format
- **Modular Design**: Each preprocessing step is a separate function
- **Configuration-Driven**: Choose steps, their order, and parameters via YAML
- **Custom Steps Support**: Extend the pipeline with your own preprocessing functions
- **Progress Tracking**: Rich progress bars show real-time progress for recordings and preprocessing steps
- **Comprehensive Logging**: MNE logger integration with optional log file output
- **Multiple Output Formats**:
  - Clean preprocessed epochs in `.fif` format
  - Clean preprocessed raw data in `.fif` format
  - Interactive HTML reports using MNE Report
  - JSON reports for easy downstream processing
- **Batch Processing**: Process multiple subjects sequentially
- **Command-line Interface**: Easy to use from the terminal

## Installation

### Option 1: Docker (Recommended)

Using Docker is the easiest way to get started, as it includes all dependencies and system libraries.

1. Build the Docker image:
```bash
git clone https://github.com/Laouen/nice-preprocessing.git
cd nice-preprocessing
docker build -t nice-preprocessing .
```

2. Run the container:
```bash
docker run --rm -v /path/to/bids/data:/data nice-preprocessing \
    --bids-root /data \
    --subjects 01 02 \
    --tasks rest \
    --config /app/configs/config_example.yaml
```

### Option 2: Local Installation

1. Clone this repository:
```bash
git clone https://github.com/Laouen/nice-preprocessing.git
cd nice-preprocessing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install the package to use the `eeg-preprocess` command:
```bash
pip install -e .
```

## Usage

### Using Docker

To use the Docker image, mount your BIDS dataset directory to `/data` in the container. The outputs will be written to the `derivatives/nice_preprocessing` subdirectory within your BIDS root.

**Basic usage:**
```bash
docker run --rm \
    -v /path/to/bids:/data \
    nice-preprocessing \
    --bids-root /data \
    --tasks rest
```

**With custom configuration:**
```bash
docker run --rm \
    -v /path/to/bids:/data \
    -v /path/to/custom/config.yaml:/config.yaml \
    nice-preprocessing \
    --bids-root /data \
    --subjects 01 02 03 \
    --tasks rest \
    --config /config.yaml
```

**With log file output:**
```bash
docker run --rm \
    -v /path/to/bids:/data \
    -v /path/to/logs:/logs \
    nice-preprocessing \
    --bids-root /data \
    --tasks rest \
    --log-file /logs/pipeline.log
```

**Processing specific sessions:**
```bash
docker run --rm \
    -v /path/to/bids:/data \
    nice-preprocessing \
    --bids-root /data \
    --subjects 01 02 \
    --sessions 01 02 \
    --tasks rest
```

### Using Local Installation

#### Process Multiple Subjects

Run the preprocessing pipeline on multiple subjects:

```bash
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 03 \
    --tasks rest \
    --config configs/config_example.yaml
```

If you installed the package with `pip install -e .`, you can use the `eeg-preprocess` command:

```bash
eeg-preprocess \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 03 \
    --tasks rest \
    --config configs/config_example.yaml
```

Process all subjects with a specific task:

```bash
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --tasks rest
```

Process specific subjects with multiple tasks:

```bash
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 \
    --tasks rest task1 task2
```

#### Python API Usage

You can also use the pipeline directly in Python:

```python
import sys
sys.path.insert(0, 'src')
from eeg_preprocessing_pipeline import EEGPreprocessingPipeline

# Load configuration
import yaml
with open('configs/config_example.yaml', 'r') as f:
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

The pipeline creates outputs in a BIDS-derivatives structure:

```
derivatives/nice_preprocessing/
├── epochs/              # When saving epochs with save_clean_instance
│   └── sub-01/
│       └── eeg/
│           └── sub-01_task-rest_proc-clean_desc-cleaned_epo.fif
├── raw/                 # When saving raw data with save_clean_instance
│   └── sub-01/
│       └── eeg/
│           └── sub-01_task-rest_proc-clean_desc-cleaned_epo.fif
└── reports/
    └── sub-01/
        └── eeg/
            ├── sub-01_task-rest_proc-clean_desc-cleaned_report.json
            └── sub-01_task-rest_proc-clean_desc-cleaned_report.html
```

### Output Details

1. **epochs/** or **raw/**: Contains MNE data objects saved in `.fif` format (if `save_clean_instance` step is included)
   - Epochs can be loaded with `mne.read_epochs()`
   - Raw data can be loaded with `mne.io.read_raw_fif()`
   - Includes all preprocessing (filtering, artifact removal, baseline correction)

2. **reports/**: Contains preprocessing reports
   - **JSON report**: Preprocessing parameters, quality metrics, steps performed (generated by `generate_json_report` step)
   - **HTML report**: Interactive visualization (generated by `generate_html_report` step)

## Configuration

The pipeline is configuration-driven. You define a list of preprocessing steps, their order, and parameters in a YAML file.

### Available Steps

- **set_montage**: Set channel montage for EEG data
- **drop_unused_channels**: Explicitly drop specified channels by name
- **bandpass_filter**: Apply bandpass filtering
- **notch_filter**: Apply notch filtering
- **resample**: Resample data to different sampling frequency
- **reference**: Apply re-referencing
- **find_flat_channels**: Find flat/disconnected channels based on variance
- **interpolate_bad_channels**: Interpolate bad channels
- **drop_bad_channels**: Drop bad channels without interpolation
- **ica**: ICA-based artifact removal
- **find_events**: Find events in the data
- **epoch**: Create epochs around events
- **find_bads_channels_threshold**: Find bad channels using threshold-based rejection
- **find_bads_channels_variance**: Find bad channels using variance-based detection
- **find_bads_channels_high_frequency**: Find bad channels using high-frequency variance
- **find_bads_epochs_threshold**: Find and remove bad epochs using threshold-based rejection
- **save_clean_instance**: Save raw or epochs data to .fif file
- **generate_json_report**: Generate JSON report
- **generate_html_report**: Generate HTML report

### Example Configuration

See `configs/config_example.yaml` for a full pipeline with epochs:

```yaml
pipeline:
  - name: bandpass_filter
    l_freq: 0.5
    h_freq: 40.0
  - name: reference
    ref_channels: average
    instance: raw
  - name: ica
    n_components: 20
    method: fastica
    find_eog: true
    find_ecg: false
    apply: true
  - name: find_events
    shortest_event: 1
  - name: epoch
    tmin: -0.2
    tmax: 0.8
    baseline: [null, 0]
    event_id: null
    reject:
      eeg: 1.5e-04
  - name: save_clean_instance
    instance: epochs
  - name: generate_json_report
  - name: generate_html_report
```

See `configs/config_raw_only.yaml` for a simpler pipeline without epoching:

```yaml
pipeline:
  - name: bandpass_filter
    l_freq: 1.0
    h_freq: 30.0
  - name: reference
    ref_channels: average
  - name: ica
    n_components: 15
    method: fastica
    find_eog: true
    apply: true
  - name: generate_json_report
```

See `configs/config_with_adaptive_reject.yaml` for a pipeline with adaptive autoreject steps. This config includes additional preprocessing steps like montage setting, notch filtering, and resampling:

```yaml
pipeline:
  - name: set_montage
    montage: standard_1020
  - name: bandpass_filter
    l_freq: 0.5
    h_freq: 45.0
  - name: notch_filter
    freqs: [50.0, 100.0]
  - name: resample
    instance: raw
    sfreq: 250.0
    npad: 'auto'
  - name: find_events
    shortest_event: 1
  - name: epoch
    tmin: -0.2
    tmax: 0.8
    baseline: [null, 0.0]
    event_id: null
    reject: null
  - name: find_bads_channels_threshold
    reject:
      eeg: 1.0e-4
    n_epochs_bad_ch: 0.5
  - name: find_bads_channels_variance
    instance: epochs
    zscore_thresh: 4
    max_iter: 2
  - name: find_bads_channels_high_frequency
    instance: epochs
    zscore_thresh: 4
    max_iter: 2
  - name: find_bads_epochs_threshold
    reject:
      eeg: 1.0e-4
    n_channels_bad_epoch: 0.1
  - name: reference
    instance: epochs
    ref_channels: average
  - name: interpolate_bad_channels
    instance: epochs
  - name: save_clean_instance
    instance: epochs
    overwrite: true
  - name: generate_json_report
  - name: generate_html_report
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
- `--log-file`: Path to log file (optional, defaults to console output)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, or ERROR (optional, default: INFO)

## Custom Preprocessing Steps

The pipeline supports custom preprocessing steps, allowing you to extend the pipeline with your own processing functions without modifying the core code.

### Creating Custom Steps

1. **Create a Python file** with your custom step functions:

```python
# my_custom_steps.py
def my_custom_filter(data, step_config):
    """Apply custom filtering to raw data."""
    if 'raw' not in data:
        raise ValueError("my_custom_filter requires 'raw' in data")
    
    # Get parameters from step_config
    cutoff_freq = step_config.get('cutoff_freq', 30.0)
    
    # Apply custom processing
    data['raw'].filter(h_freq=cutoff_freq, l_freq=None)
    
    # Record the step for reporting
    data['preprocessing_steps'].append({
        'step': 'my_custom_filter',
        'cutoff_freq': cutoff_freq
    })
    
    return data
```

2. **Place the file in a dedicated folder**, for example: `/path/to/my_custom_steps/`

3. **Update your config file** to specify the custom steps folder:

```yaml
custom_steps_folder: /path/to/my_custom_steps

pipeline:
  - name: my_custom_filter
    cutoff_freq: 30.0
  - name: bandpass_filter  # Built-in steps still work
    l_freq: 0.5
    h_freq: 40.0
```

4. **Run the pipeline** as usual - custom steps are automatically loaded and available.

### Custom Step Requirements

Custom step functions must follow these rules:

- **Signature**: Accept exactly 2 parameters: `data` (Dict) and `step_config` (Dict)
- **Return**: Return the updated `data` dictionary
- **Validation**: Check that required data instances exist (e.g., `'raw'`, `'epochs'`)
- **Recording**: Append a summary to `data['preprocessing_steps']` for reporting
- **Naming**: Function names become step names; avoid starting with underscore

See `configs/example_custom_steps.py` for complete examples.

### Using Custom Steps with Docker

Mount your custom steps folder when running the container:

```bash
docker run -v /host/bids:/data \
           -v /host/custom_steps:/custom_steps \
           -v /host/config:/config \
           nice-preprocessing \
           --bids-root /data \
           --subjects 01 02 \
           --tasks rest \
           --config /config/my_config.yaml
```

In your config file, use the container path:

```yaml
custom_steps_folder: /custom_steps
pipeline:
  - name: my_custom_filter
    cutoff_freq: 30.0
```

### Advanced Features

- **Override built-in steps**: Custom steps with the same name as built-in steps will override them
- **Multiple files**: Place multiple `.py` files in the custom steps folder - all will be loaded
- **Error handling**: If a custom step file has errors, other files will still be loaded
- **Private functions**: Functions starting with `_` are ignored and not loaded as steps

## Preprocessing Steps Details

Each step can be customized through the configuration:

### Excluding Channels from Analysis

Many preprocessing steps support an `excluded_channels` parameter that allows you to exclude specific channels (e.g., reference channels like 'Cz') from analysis to avoid reference problems. This is useful when you want to preserve a reference channel or exclude channels that should not be analyzed in certain steps.

**Steps that support `excluded_channels`:**
- `bandpass_filter` - Exclude channels from filtering
- `notch_filter` - Exclude channels from notch filtering
- `ica` - Exclude channels from ICA decomposition
- `find_flat_channels` - Exclude channels from flat channel detection
- `find_bads_channels_threshold` - Exclude channels from bad channel detection
- `find_bads_channels_variance` - Exclude channels from variance-based detection
- `find_bads_channels_high_frequency` - Exclude channels from high-frequency analysis
- `find_bads_epochs_threshold` - Exclude channels from epoch rejection criteria
- `interpolate_bad_channels` - Exclude channels from interpolation even if marked as bad
- `drop_bad_channels` - Exclude channels from dropping even if marked as bad

**Steps where exclusion doesn't apply:**
- `reference` - Reference computation uses selected channels; use `ref_channels` parameter instead
- `resample` - Resamples all data uniformly
- `set_montage` - Sets electrode positions for all channels
- `drop_unused_channels` - Use this for explicit channel removal

**Example usage:**
```yaml
- name: bandpass_filter
  l_freq: 0.5
  h_freq: 45.0
  excluded_channels: ['Cz']  # Exclude Cz from filtering

- name: find_bads_channels_threshold
  reject:
    eeg: 1.0e-4
  excluded_channels: ['Cz', 'FCz']  # Don't mark these as bad

- name: drop_bad_channels
  instance: epochs
  excluded_channels: ['Cz']  # Don't drop Cz even if marked as bad
```

See `configs/config_with_excluded_channels.yaml` for a complete example.

### 1. set_montage
Set channel montage for EEG data. Useful when data lacks electrode position information.
- `montage`: Name of standard montage to use (default: 'standard_1020')
  - Examples: 'standard_1020', 'standard_1005', 'biosemi64', etc.
  - See MNE documentation for available montages

### 2. drop_unused_channels
Explicitly drop specified channels from the data by name. Different from drop_bad_channels, this drops channels regardless of whether they're marked as bad.
- `channels_to_drop`: List of channel names to drop
- `instance`: Which data instance to drop channels from - 'raw' or 'epochs' (default: 'raw')

### 3. bandpass_filter
Apply bandpass filtering.
- `l_freq`: High-pass filter frequency (Hz)
- `h_freq`: Low-pass filter frequency (Hz)
- `l_freq_order`: Filter order for high-pass (default: 6)
- `h_freq_order`: Filter order for low-pass (default: 8)
- `picks`: Optional channel indices to filter
- `excluded_channels`: List of channel names to exclude from filtering (optional)
- `n_jobs`: Number of parallel jobs (default: 1)

### 4. notch_filter
Apply notch filtering to remove line noise.
- `freqs`: Frequencies to notch filter (e.g., [50.0, 100.0])
- `notch_widths`: Width of notch filters (optional)
- `method`: Filtering method (default: 'fft')
- `picks`: Optional channel indices to filter
- `excluded_channels`: List of channel names to exclude from filtering (optional)
- `n_jobs`: Number of parallel jobs (default: 1)

### 5. resample
Resample the data to a different sampling frequency.
- `instance`: Which data instance to resample - 'raw' or 'epochs' (default: 'raw')
- `sfreq`: Target sampling frequency in Hz (default: 250)
- `npad`: Padding to use for resampling (default: 'auto')
- `resample_events`: Whether to also resample events (default: false)
- `n_jobs`: Number of parallel jobs (default: 1)

### 6. reference
Apply re-referencing.
- `ref_channels`: Reference channels ('average' or channel names)
- `instance`: Which data instance to reference - 'raw' or 'epochs' (default: 'epochs')

### 7. find_flat_channels
Find flat/disconnected channels based on variance threshold. Channels with variance below the threshold are marked as bad.
- `picks`: Channel indices to check (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from flat channel detection (optional)
- `threshold`: Variance threshold below which channels are considered flat (default: 1e-12)

### 8. interpolate_bad_channels
Interpolate bad channels using spherical spline interpolation.
- `instance`: Which data instance to interpolate - 'raw' or 'epochs' (default: 'epochs')
- `excluded_channels`: List of channel names to exclude from interpolation (optional)

### 9. drop_bad_channels
Drop bad channels without interpolation. This step removes channels marked as bad from the data instead of interpolating them.
- `instance`: Which data instance to drop channels from - 'raw' or 'epochs' (default: 'epochs')
- `excluded_channels`: List of channel names to exclude from dropping even if marked as bad (optional)

### 10. ica
ICA-based artifact removal.
- `n_components`: Number of ICA components (default: 20)
- `method`: ICA method ('fastica', 'infomax', 'picard', default: 'fastica')
- `random_state`: Random state for reproducibility (default: 97)
- `excluded_channels`: List of channel names to exclude from ICA decomposition (optional)
- `find_eog`: Automatically find EOG artifacts (true/false, default: true)
- `find_ecg`: Automatically find ECG artifacts (true/false, default: false)
- `apply`: Apply ICA to remove artifacts (true/false, default: true)

### 11. find_events
Find events in the data.
- `shortest_event`: Minimum event duration in samples (default: 1)

### 12. epoch
Create epochs around events.
- `tmin`: Start time before event (seconds, default: -0.2)
- `tmax`: End time after event (seconds, default: 0.5)
- `baseline`: Baseline correction window (tuple or null, default: (null, 0.0))
- `event_id`: Event IDs to include (dict or null for all)
- `reject`: Rejection criteria (dict with channel type keys, optional)

### 13. find_bads_channels_threshold
Find bad channels using threshold-based rejection. Marks channels as bad if they exceed rejection thresholds in too many epochs.
- `picks`: Channel indices to check (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from bad channel detection (optional)
- `reject`: Rejection thresholds by channel type (e.g., `{"eeg": 150e-6}`)
- `n_epochs_bad_ch`: Fraction or number of epochs a channel must be bad in to be marked as bad (default: 0.5)
- `apply_on`: List of instances to mark bad channels on (default: ['epochs'])

### 14. find_bads_channels_variance
Find bad channels using variance-based detection. Identifies channels with abnormally high or low variance.
- `instance`: Which data instance to use - 'raw' or 'epochs' (default: 'epochs')
- `picks`: Channel indices to check (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from variance analysis (optional)
- `zscore_thresh`: Z-score threshold for outlier detection (default: 4)
- `max_iter`: Maximum iterations for iterative outlier removal (default: 2)
- `apply_on`: List of instances to mark bad channels on (default: [instance])

### 15. find_bads_channels_high_frequency
Find bad channels using high-frequency variance. Detects channels with excessive high-frequency noise.
- `instance`: Which data instance to use - 'raw' or 'epochs' (default: 'epochs')
- `picks`: Channel indices to check (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from high-frequency analysis (optional)
- `zscore_thresh`: Z-score threshold for outlier detection (default: 4)
- `max_iter`: Maximum iterations for iterative outlier removal (default: 2)
- `apply_on`: List of instances to mark bad channels on (default: [instance])

### 16. find_bads_epochs_threshold
Find and remove bad epochs using threshold-based rejection. Drops epochs that have too many bad channels.
- `picks`: Channel indices to check (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from epoch rejection criteria (optional)
- `reject`: Rejection thresholds by channel type (e.g., `{"eeg": 150e-6}`)
- `n_channels_bad_epoch`: Fraction or number of channels that must be bad for an epoch to be rejected (default: 0.1)

### 17. save_clean_instance
Save clean raw or epochs data to .fif file in BIDS-derivatives format.
- `instance`: Which data instance to save - 'raw' or 'epochs' (default: 'epochs')
- `overwrite`: Whether to overwrite existing files (default: true)

### 18. generate_json_report
Generate JSON report with preprocessing information. No parameters needed.

### 19. generate_html_report
Generate HTML report with interactive visualizations. No parameters needed.

## Batch Processing

The pipeline processes multiple subjects and files sequentially. You can process:

```bash
# Process specific subjects with a specific task
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 03 04 05 \
    --tasks rest \
    --config configs/config_example.yaml

# Process all subjects in the dataset
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --config configs/config_example.yaml

# Process specific sessions for specific subjects
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 \
    --sessions 01 02 \
    --tasks rest
```

For HPC/cluster environments, you can create your own SLURM or other batch submission scripts that call the pipeline with subject lists.

## Progress Tracking and Logging

The pipeline includes comprehensive progress tracking and logging features:

### Progress Bars

When running the pipeline, you'll see two levels of progress bars:
1. **Overall progress**: Shows progress across all recordings being processed
2. **Step progress**: Shows progress through preprocessing steps for each recording

The progress bars use the `rich` library and display:
- Spinner animation
- Progress bar with percentage
- Time remaining estimate
- Current step being executed

### Logging

The pipeline uses MNE's logger for all output messages. You can:

**Console Output (default)**:
```bash
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02
```

**Log to File**:
```bash
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 \
    --log-file /path/to/logs/pipeline.log
```

**Adjust Logging Level**:
```bash
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 \
    --log-level DEBUG
```

Available log levels: `DEBUG`, `INFO` (default), `WARNING`, `ERROR`

The pipeline also saves a summary of results to `derivatives/nice_preprocessing/pipeline_results.json` for easy programmatic access.

## Docker Notes

### Volume Mounting

When using Docker, you need to mount your local directories to paths inside the container using the `-v` flag:

- **BIDS dataset**: Mount your BIDS root directory to `/data` or any path you specify with `--bids-root`
- **Configuration files**: Mount custom config files if not using the built-in configs in `/app/configs/`
- **Output directory**: The pipeline writes outputs to `<bids-root>/derivatives/nice_preprocessing/` by default
- **Log files**: If using `--log-file`, mount a directory for log output

### File Permissions

The Docker container runs as root by default. Files created by the container will be owned by root. To avoid permission issues:

1. Run with your user ID:
```bash
docker run --rm --user $(id -u):$(id -g) \
    -v /path/to/bids:/data \
    nice-preprocessing \
    --bids-root /data \
    --tasks rest
```

2. Or fix permissions after processing:
```bash
sudo chown -R $USER:$USER /path/to/bids/derivatives
```

### Using Built-in Configurations

The Docker image includes several pre-configured pipeline examples in `/app/configs/`:
- `/app/configs/config_example.yaml` - Standard pipeline with epochs
- `/app/configs/config_raw_only.yaml` - Raw data processing without epoching
- `/app/configs/config_with_adaptive_reject.yaml` - Advanced pipeline with adaptive artifact rejection
- `/app/configs/config_minimal.yaml` - Minimal preprocessing steps

Example using a built-in config:
```bash
docker run --rm \
    -v /path/to/bids:/data \
    nice-preprocessing \
    --bids-root /data \
    --tasks rest \
    --config /app/configs/config_with_adaptive_reject.yaml
```

### Building from Source

If you want to customize the Docker image or use a development version:

```bash
git clone https://github.com/Laouen/nice-preprocessing.git
cd nice-preprocessing
docker build -t nice-preprocessing:custom .
```

**Building in CI/CD environments with self-signed certificates:**

If you're building in a CI/CD environment with self-signed SSL certificates, use the `PIP_TRUSTED_HOST` build argument:

```bash
docker build --build-arg PIP_TRUSTED_HOST=1 -t nice-preprocessing:custom .
```

Note: This disables SSL verification for PyPI and should only be used in trusted CI/CD environments, not for production builds.

## Requirements

- Python >= 3.8
- mne >= 1.5.0
- mne-bids >= 0.14
- numpy >= 1.24.0
- scipy >= 1.11.0
- rich >= 13.0.0
- matplotlib >= 3.7.0 (recommended)
- pandas >= 2.0.0 (recommended)

## License

This project is ready to use for several projects and includes scripts for SLURM execution.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on the GitHub repository. 
