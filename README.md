# NICE EEG Preprocessing Pipeline

A modular, configuration-driven EEG preprocessing pipeline using MNE-BIDS. The pipeline uses auxiliary functions for each preprocessing step, allowing you to choose which steps to run, their order, and their parameters through a simple YAML configuration.

## Architecture

The pipeline is split into two independent modules that can run separately or together:

### 🔧 Preprocessing Module
Handles all EEG data preprocessing steps (filtering, ICA, bad channel detection, epoching, etc.) and saves intermediate results. See [README_preprocessing.md](README_preprocessing.md) for details.

### 📊 Report Module
Generates JSON and HTML reports from saved intermediate results. Can run independently after preprocessing is complete. See [README_report.md](README_report.md) for details.

### 🔄 Combined Pipeline (Original Behavior)
The original combined pipeline (`eeg-preprocess` command) is still available for backward compatibility.

## Features

- **Flexible File Discovery**: Support for both BIDS-formatted datasets and custom glob patterns
- **MNE-BIDS Integration**: Seamlessly reads EEG data in BIDS format
- **Modular Design**: Each preprocessing step is a separate function
- **Split Architecture**: Run preprocessing and reporting separately or together
- **Intermediate Results**: Save and reuse preprocessed data
- **Configuration-Driven**: Choose steps, their order, and parameters via YAML
- **Custom Steps Support**: Extend the pipeline with your own preprocessing functions
- **Progress Tracking**: Rich progress bars show real-time progress for recordings and preprocessing steps
- **Comprehensive Logging**: MNE logger integration with optional log file output
- **Multiple Output Formats**:
  - Clean preprocessed epochs in `.fif` format
  - Clean preprocessed raw data in `.fif` format
  - Intermediate results (pickled data objects)
  - Interactive HTML reports using MNE Report
  - JSON reports for easy downstream processing
- **Batch Processing**: Process multiple subjects sequentially
- **Multiple Command-line Interfaces**: 
  - `eeg-preprocess` - Combined preprocessing + reporting (original)
  - `eeg-preprocess-only` - Preprocessing only with intermediate results
  - `eeg-generate-reports` - Report generation from intermediate results
- **Docker Support**: Separate containers for preprocessing and reporting
- **Docker Compose**: Orchestrate both modules together

## Installation

### Option 1: Docker (Recommended)

Using Docker is the easiest way to get started, as it includes all dependencies and system libraries.

#### Option 1a: Docker Compose (Easiest)

```bash
git clone https://github.com/Laouen/nice-preprocessing.git
cd nice-preprocessing

# Set your data directory
export BIDS_ROOT=/path/to/bids
export CONFIG_DIR=$(pwd)/configs

# Build all containers
docker-compose build

# Run preprocessing only
docker-compose run --rm preprocessing \
    --bids-root /data \
    --config /configs/config_example.yaml

# Generate reports
docker-compose run --rm report \
    --bids-root /data \
    --all

# Or run combined pipeline (original behavior)
docker-compose run --rm combined \
    --bids-root /data \
    --config /configs/config_example.yaml
```

#### Option 1b: Individual Docker Images

Build and use specific images:

```bash
# Combined pipeline (original)
docker build -t nice-preprocessing .
docker run --rm -v /path/to/bids:/data nice-preprocessing \
    --bids-root /data \
    --config /app/configs/config_example.yaml

# Preprocessing only
docker build -f Dockerfile.preprocessing -t nice-preprocessing-module .
docker run --rm -v /path/to/bids:/data nice-preprocessing-module \
    --bids-root /data \
    --config /app/configs/config_example.yaml

# Report generation
docker build -f Dockerfile.report -t nice-report-generator .
docker run --rm -v /path/to/bids:/data nice-report-generator \
    --bids-root /data \
    --all
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

The pipeline now supports three workflow modes:

### 1. Split Workflow (Recommended for Flexibility)

Run preprocessing and report generation separately:

```bash
# Step 1: Run preprocessing with intermediate results
eeg-preprocess-only \
    --bids-root /path/to/bids \
    --config configs/config_with_save.yaml

# Step 2: Generate reports (can be done later, on different machine, etc.)
eeg-generate-reports \
    --bids-root /path/to/bids \
    --all
```

**Benefits:**
- Generate reports multiple times without reprocessing
- Run preprocessing on HPC, generate reports locally
- Save time when experimenting with report parameters
- Can delete intermediate results after verification to save space

See [README_preprocessing.md](README_preprocessing.md) and [README_report.md](README_report.md) for detailed documentation.

### 2. Combined Workflow (Original Behavior)

Run preprocessing and reporting together in one command:

```bash
eeg-preprocess \
    --bids-root /path/to/bids \
    --config configs/config_example.yaml
```

**Benefits:**
- Single command workflow
- Backward compatible with existing scripts
- No intermediate results storage needed

### 3. Docker Compose Workflow

Use docker-compose for orchestrated multi-container setup:

```bash
# Set environment variables
export BIDS_ROOT=/path/to/bids
export CONFIG_DIR=$(pwd)/configs

# Run preprocessing
docker-compose run --rm preprocessing \
    --bids-root /data \
    --config /configs/config_with_save.yaml

# Generate reports
docker-compose run --rm report \
    --bids-root /data \
    --all
```

## Quick Start Examples

### Example 1: Complete Split Workflow
```bash
docker run --rm \
    -v /path/to/bids:/data \
    nice-preprocessing \
    --bids-root /data \
    --subjects 01 02 \
    --sessions 01 02 \
    --tasks rest
```


```bash
# Preprocessing with specific subjects
eeg-preprocess-only \
    --bids-root /path/to/bids \
    --subjects 01 02 03 \
    --tasks rest \
    --config configs/config_with_save.yaml

# Generate reports for those subjects
eeg-generate-reports \
    --bids-root /path/to/bids \
    --subjects 01 02 03
```

### Example 2: Combined Workflow (Original)

```bash
eeg-preprocess \
    --bids-root /path/to/bids \
    --subjects 01 02 03 \
    --tasks rest \
    --config configs/config_example.yaml
```

### Example 3: Docker Workflow

```bash
# Preprocessing
docker run --rm -v /path/to/bids:/data \
    nice-preprocessing-module \
    --bids-root /data \
    --config /app/configs/config_with_save.yaml

# Report generation
docker run --rm -v /path/to/bids:/data \
    nice-report-generator \
    --bids-root /data \
    --all
```

## Configuration

### For Split Workflow

Your configuration file must include the `save_intermediate_results` step:

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
  
  # Save intermediate results for later report generation
  - name: save_intermediate_results
    save_raw: true
    save_epochs: true
    save_ica: true
```

### For Combined Workflow

Include report generation steps in your configuration:

```yaml
pipeline:
  - name: bandpass_filter
    l_freq: 0.5
    h_freq: 40.0
  
  # ... other preprocessing steps ...
  
  - name: save_clean_instance
    instance: epochs
  
  - name: generate_json_report
  
  - name: generate_html_report
```

See `configs/` directory for more examples.

## Module Documentation

For detailed documentation on each module:

- **[README_preprocessing.md](README_preprocessing.md)** - Complete guide to the preprocessing module
  - All preprocessing steps and their parameters
  - Intermediate results format
  - Command-line arguments
  - Docker usage
  - Python API examples

- **[README_report.md](README_report.md)** - Complete guide to the report generation module
  - Report contents and formats
  - Command-line arguments
  - Docker usage
  - Python API examples
  - Workflow patterns

- **[SPLIT_MODULES_DOCUMENTATION.md](SPLIT_MODULES_DOCUMENTATION.md)** - Technical documentation on the split architecture
  - Design decisions
  - API reference
  - Migration guide
  - Troubleshooting

## Available Commands

After installation (`pip install -e .`), you have access to three CLI commands:

| Command | Purpose | Module |
|---------|---------|--------|
| `eeg-preprocess` | Combined preprocessing + reporting (original) | Combined |
| `eeg-preprocess-only` | Preprocessing with intermediate results | Preprocessing |
| `eeg-generate-reports` | Report generation from intermediate results | Report |

## Docker Images

Three Docker images are available:

| Image | Dockerfile | Purpose | Entry Point |
|-------|-----------|---------|-------------|
| `nice-preprocessing` | `Dockerfile` | Combined pipeline | `eeg-preprocess` |
| `nice-preprocessing-module` | `Dockerfile.preprocessing` | Preprocessing only | `eeg-preprocess-only` |
| `nice-report-generator` | `Dockerfile.report` | Report generation | `eeg-generate-reports` |


```
bids_root/derivatives/nice_preprocessing/
├── intermediate/                           # Intermediate results (split workflow)
│   └── sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}/
│       ├── metadata.json                   # Preprocessing steps and parameters
│       ├── raw.pkl                         # Raw data object
│       ├── epochs.pkl                      # Epochs object
│       ├── ica.pkl                         # ICA object
│       └── events.pkl                      # Events array
│
├── raw/                                    # Clean raw data (.fif files)
│   └── sub-{subject}/[ses-{session}/]eeg/
│       └── sub-{subject}_[ses-{session}_]task-{task}_[acq-{acquisition}_]
│           processing-clean_desc-cleaned_epo.fif
│
├── epochs/                                 # Clean epochs (.fif files)
│   └── sub-{subject}/[ses-{session}/]eeg/
│       └── sub-{subject}_[ses-{session}_]task-{task}_[acq-{acquisition}_]
│           processing-clean_desc-cleaned_epo.fif
│
└── reports/                                # JSON and HTML reports
    └── sub-{subject}/[ses-{session}/]eeg/
        ├── sub-{subject}_[ses-{session}_]task-{task}_[acq-{acquisition}_]
        │   processing-clean_desc-cleaned_report.json
        └── sub-{subject}_[ses-{session}_]task-{task}_[acq-{acquisition}_]
            processing-clean_desc-cleaned_report.html
```

## Python API Usage

### Combined Pipeline (Original)

```python
from eeg_preprocessing_pipeline import EEGPreprocessingPipeline
import yaml

# Load configuration
with open('configs/config_example.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize and run
pipeline = EEGPreprocessingPipeline(
    bids_root='/path/to/bids',
    config=config
)
results = pipeline.run_pipeline(subjects=['01', '02'], tasks='rest')
```

### Split Workflow - Preprocessing

```python
from preprocessing_pipeline import PreprocessingPipeline
import yaml

# Load configuration (with save_intermediate_results step)
with open('configs/config_with_save.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run preprocessing
pipeline = PreprocessingPipeline(
    bids_root='/path/to/bids',
    config=config
)
results = pipeline.run_pipeline(subjects=['01', '02'], tasks='rest')
```

### Split Workflow - Report Generation

```python
from report_generator import ReportGenerator

# Initialize report generator
report_gen = ReportGenerator(bids_root='/path/to/bids')

# Generate reports for all intermediate results
report_gen.generate_all_reports()

# Or for specific recording
report_gen.generate_reports(
    subject='01',
    session='01',
    task='rest',
    acquisition=None
)
```

## Available Preprocessing Steps

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
from readers import BIDSReader

# Load configuration
import yaml
with open('configs/config_example.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create a BIDS reader
reader = BIDSReader('/path/to/bids/dataset')

# Initialize pipeline
pipeline = EEGPreprocessingPipeline(
    reader=reader,
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

## File Discovery with Readers

The pipeline supports two types of file readers for discovering data files:

### BIDS Reader (Default)

The BIDS reader uses MNE-BIDS to automatically discover files in BIDS-formatted datasets:

```bash
# BIDS reader is the default (--reader bids can be omitted)
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 \
    --tasks rest \
    --config configs/config_example.yaml
```

### Glob Reader

The glob reader allows you to work with custom directory structures using glob patterns with variable extraction:

```bash
python src/cli.py \
    --reader glob \
    --data-root /path/to/data \
    --glob-pattern "sub-{subject}/ses-{session}/eeg/sub-{subject}_task-{task}_eeg.vhdr" \
    --subjects 01 02 \
    --tasks rest \
    --config configs/config_example.yaml
```

**Pattern syntax:** Use `{variable_name}` placeholders which:
- Convert to `*` wildcards for file matching
- Extract matched values as metadata

**Python API:**

```python
from readers import GlobReader

# Create a glob reader with your custom pattern
reader = GlobReader(
    data_root='/path/to/data',
    pattern='sub-{subject}/ses-{session}/eeg/sub-{subject}_task-{task}_eeg.vhdr'
)

# Initialize pipeline with the glob reader
pipeline = EEGPreprocessingPipeline(
    reader=reader,
    config=config
)

# Run pipeline
results = pipeline.run_pipeline(subjects=['01', '02'], tasks='rest')
```

For detailed information on readers, pattern examples, and troubleshooting, see [READERS.md](READERS.md).

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

Data Organization:
- **strip_recording**: Crop recordings to remove data outside the first and last events
- **concatenate_recordings**: Concatenate multiple raw recordings into a single continuous recording
- **copy_instance**: Create a copy of a data instance for comparison or backup purposes

Setup:
- **set_montage**: Set channel montage for EEG data
- **drop_unused_channels**: Explicitly drop specified channels by name

Filtering:
- **bandpass_filter**: Apply bandpass filtering
- **notch_filter**: Apply notch filtering

Preprocessing:
- **resample**: Resample data to different sampling frequency
- **reference**: Apply re-referencing
- **ica**: ICA-based artifact removal

Bad Channel Detection:
- **find_flat_channels**: Find flat/disconnected channels based on variance
- **find_bads_channels_threshold**: Find bad channels using threshold-based rejection
- **find_bads_channels_variance**: Find bad channels using variance-based detection
- **find_bads_channels_high_frequency**: Find bad channels using high-frequency variance

Bad Channel Handling:
- **interpolate_bad_channels**: Interpolate bad channels
- **drop_bad_channels**: Drop bad channels without interpolation

Epoching:
- **find_events**: Find events in the data
- **epoch**: Create epochs around events
- **chunk_in_epoch**: Create fixed-length epochs from continuous data
- **find_bads_epochs_threshold**: Find and remove bad epochs using threshold-based rejection

Output:
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
    instance: 'raw'
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
  - name: concatenate_recordings
  
  - name: set_montage
    montage: standard_1020
  
  - name: bandpass_filter
    l_freq: 0.1
    h_freq: 40.0
  
  - name: notch_filter
    freqs: [50.0, 100.0]
  
  - name: resample
    instance: raw
    sfreq: 250.0
    npad: auto
  
  - name: find_events
    get_events_from: annotations
    shortest_event: 1
    event_id:
      stim/12hz: 10001
      stim/15hz: 10002
  
  - name: epoch
    tmin: -0.2
    tmax: 1.2
    baseline: [null, 0.0]
    reject: null
  
  - name: reference
    instance: 'epochs'
    ref_channels: average
  
  - name: reference
    instance: 'raw'
    ref_channels: average
  
  - name: generate_html_report
```

Note: This config file also includes commented-out examples of bad channel detection steps (find_bads_channels_threshold, find_bads_channels_variance, find_bads_channels_high_frequency) that can be uncommented and customized as needed.

See `configs/config_minimal.yaml` for a comprehensive pipeline including strip_recording, copy_instance, and ICA:

```yaml
pipeline:
  - name: strip_recording
    instance: all_raw
    get_events_from: annotations
    shortest_event: 5
    start_padding: 1
    end_padding: 1
  
  - name: concatenate_recordings
  
  - name: set_montage
    montage: GSN-HydroCel-256
  
  - name: copy_instance
    from_instance: raw
    to_instance: raw_before_cleaning
  
  - name: find_flat_channels
    threshold: 1.0e-12
  
  - name: bandpass_filter
    l_freq: 0.1
    h_freq: 40.0
  
  - name: chunk_in_epoch
    duration: 1
  
  - name: ica
    n_components: 20
    method: fastica
    find_eog: true
    apply: true
  
  - name: save_clean_instance
    instance: epochs
    overwrite: true
  
  - name: generate_html_report
    compare_instances:
      - title: 'Before vs After Cleaning'
        instance_a:
          name: 'raw'
          label: 'After Cleaning'
        instance_b:
          name: 'raw_before_cleaning'
          label: 'Before Cleaning'
```

Additional example configurations available in `configs/`:
- `config_with_drop_bad_channels.yaml` - Example using drop_bad_channels instead of interpolation
- `config_with_excluded_channels.yaml` - Example using excluded_channels parameter to preserve reference channels
- `config_with_custom_steps.yaml` - Example showing how to integrate custom preprocessing steps

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
- `--extension`: File extension to process (default: `.vhdr`)

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

### Data Organization Steps

#### strip_recording
Crop recordings to remove data outside the first and last events. This is useful for removing unnecessary data at the beginning and end of recordings that don't contain task-relevant data.
- `instance`: Which data instance to crop - 'all_raw' or 'raw' (default: 'raw')
- `get_events_from`: How to extract events - 'stim' or 'annotations' (default: 'annotations')
- `shortest_event`: Minimum number of samples for an event (default: 1)
- `event_id`: Event IDs to use for finding start/end points. Can be a dict mapping event names to IDs or 'auto' (default: 'auto')
- `start_padding`: Time in seconds to keep before the first event (default: 1)
- `end_padding`: Time in seconds to keep after the last event (default: 1)

**Example:**
```yaml
- name: strip_recording
  instance: all_raw
  get_events_from: annotations
  shortest_event: 1
  event_id:
    Stimulus/CatNewRepeated/CR: 91
    Stimulus/CatOld/Hit: 101
  start_padding: 1.0
  end_padding: 1.0
```

#### concatenate_recordings
Concatenate multiple raw recordings into a single continuous recording. This is useful when data is split across multiple files but needs to be processed as a single session.
- No parameters required
- Requires 'all_raw' to be present in data
- Creates a single 'raw' instance from all recordings in 'all_raw'

**Example:**
```yaml
- name: concatenate_recordings
```

#### copy_instance
Create a copy of a data instance. This is useful for comparing data at different stages of preprocessing (e.g., before/after cleaning or ICA).
- `from_instance`: Name of the instance to copy from (default: 'raw')
- `to_instance`: Name of the new instance to create (default: 'raw_cleaned')

**Example:**
```yaml
- name: copy_instance
  from_instance: raw
  to_instance: raw_before_ica
```

### Preprocessing Steps

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
- `picks`: Channel types to include in ICA (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from ICA decomposition (optional)
- `ica_fit_l_freq`: High-pass frequency for filtering data before ICA fit (default: 1.0 Hz)
- `ica_fit_h_freq`: Low-pass frequency for filtering data before ICA fit (optional, default: None)
- `find_eog`: Automatically find EOG artifacts (true/false, default: false)
  - `eog_channels`: List of channel names to use for EOG detection (optional, auto-detects if not provided)
  - `eog_threshold`: Correlation threshold for EOG component detection (default: 'auto')
  - `eog_measure`: Measure for EOG detection ('correlation' or 'ctps', default: 'correlation')
  - `eog_l_freq`: High-pass frequency for EOG correlation (default: 1.0 Hz)
  - `eog_h_freq`: Low-pass frequency for EOG correlation (default: 10.0 Hz)
- `find_ecg`: Automatically find ECG artifacts (true/false, default: false)
  - `ecg_channels`: List of channel names to use for ECG detection (optional)
  - `ecg_threshold`: Correlation threshold for ECG component detection (default: 'auto')
  - `ecg_measure`: Measure for ECG detection ('correlation' or 'ctps', default: 'correlation')
  - `ecg_l_freq`: High-pass frequency for ECG correlation (default: 1.0 Hz)
  - `ecg_h_freq`: Low-pass frequency for ECG correlation (default: 10.0 Hz)
- `selected_indices`: Manually specify component indices to exclude (optional, list of integers)
- `apply`: Apply ICA to remove artifacts (true/false, default: true)

### 11. find_events
Find events in the data.
- `get_events_from`: How to extract events - 'stim' or 'annotations' (default: 'annotations')
- `shortest_event`: Minimum event duration in samples (default: 1)
- `event_id`: Event IDs to extract. Can be 'auto' for all events or a dict mapping event names to IDs (default: 'auto')

### 12. epoch
Create epochs around events.
- `tmin`: Start time before event (seconds, default: -0.2)
- `tmax`: End time after event (seconds, default: 0.5)
- `baseline`: Baseline correction window (tuple or null, default: (null, 0.0))
- `event_id`: Event IDs to include (dict or null for all)
- `reject`: Rejection criteria (dict with channel type keys, optional)

### 13. chunk_in_epoch
Create fixed-length epochs from continuous raw data. This is an alternative to event-based epoching that splits the data into equal-duration segments.
- `duration`: Duration of each epoch in seconds (default: 1.0)

**Example:**
```yaml
- name: chunk_in_epoch
  duration: 1.0  # Create 1-second epochs
```

### 14. find_bads_channels_threshold
Find bad channels using threshold-based rejection. Marks channels as bad if they exceed rejection thresholds in too many epochs.
- `picks`: Channel indices to check (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from bad channel detection (optional)
- `reject`: Rejection thresholds by channel type (e.g., `{"eeg": 150e-6}`)
- `n_epochs_bad_ch`: Fraction or number of epochs a channel must be bad in to be marked as bad (default: 0.5)
- `apply_on`: List of instances to mark bad channels on (default: ['epochs'])

### 15. find_bads_channels_variance
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
Generate HTML report with interactive visualizations.
- `picks`: Channel types to include in plots (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from plots (optional)
- `compare_instances`: List of instance comparisons to plot (optional, see config_minimal.yaml for example)
- `plot_raw_kwargs`: Additional keyword arguments for raw data plots (optional, dict)
- `plot_ica_kwargs`: Additional keyword arguments for ICA plots (optional, dict)
- `plot_events_kwargs`: Additional keyword arguments for event plots (optional, dict)
- `plot_epochs_kwargs`: Additional keyword arguments for epoch plots (optional, dict)
- `plot_evokeds_kwargs`: Additional keyword arguments for evoked response plots (optional, dict)

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
- `/app/configs/config_with_adaptive_reject.yaml` - Advanced pipeline with concatenation and event-based epochs
- `/app/configs/config_minimal.yaml` - Comprehensive pipeline with strip_recording, ICA, and instance comparison
- `/app/configs/config_with_drop_bad_channels.yaml` - Pipeline using drop_bad_channels instead of interpolation
- `/app/configs/config_with_excluded_channels.yaml` - Pipeline demonstrating excluded_channels parameter
- `/app/configs/config_with_custom_steps.yaml` - Example template for using custom preprocessing steps

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
