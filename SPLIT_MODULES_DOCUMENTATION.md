# Split Preprocessing and Report Generation Modules

## Overview

The EEG preprocessing pipeline has been split into two independent modules:

1. **Preprocessing Module** (`src/preprocessing_pipeline.py`) - Handles all preprocessing steps
2. **Report Generation Module** (`src/report_generator.py`) - Generates JSON and HTML reports from saved results

This separation allows you to:
- Run preprocessing and report generation independently
- Generate reports multiple times without re-running preprocessing
- Process data on one machine and generate reports on another
- Customize report generation parameters without reprocessing

## Module Details

### 1. Preprocessing Module

**File:** `src/preprocessing_pipeline.py`  
**Class:** `PreprocessingPipeline`  
**CLI:** `src/preprocessing_cli.py`  
**Console Command:** `eeg-preprocess-only`

**Features:**
- Contains all preprocessing steps (filtering, ICA, epoching, etc.)
- New step: `save_intermediate_results` to persist data and metadata
- Saves to: `bids_root/derivatives/nice_preprocessing/intermediate/<recording_id>/`

**Saved Data:**
- `metadata.json` - Contains subject, task, session, preprocessing steps, events
- `raw.pkl` - MNE Raw object (if present)
- `epochs.pkl` - MNE Epochs object (if present)
- `ica.pkl` - MNE ICA object (if present)

**Usage Example:**
```bash
# Run preprocessing only
eeg-preprocess-only --bids-root /data/bids --config config.yaml --subjects 01

# Or with Python API
from preprocessing_pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline(bids_root='/data/bids', config=config)
results = pipeline.run_pipeline(subjects=['01'], tasks='rest')
```

**Config Example:**
```yaml
pipeline:
  - name: bandpass_filter
    l_freq: 0.5
    h_freq: 40.0
  - name: ica
    n_components: 20
  - name: epoch
    tmin: -0.2
    tmax: 0.5
  - name: save_intermediate_results  # Save results for later report generation
```

### 2. Report Generation Module

**File:** `src/report_generator.py`  
**Class:** `ReportGenerator`  
**CLI:** `src/report_cli.py`  
**Console Command:** `eeg-generate-reports`

**Features:**
- Loads intermediate results from disk
- Generates JSON reports with preprocessing summary
- Generates HTML reports with visualizations
- Can be run independently from preprocessing

**Usage Examples:**

```bash
# Generate reports for all saved intermediate results
eeg-generate-reports --bids-root /data/bids --all

# Generate reports for specific subjects
eeg-generate-reports --bids-root /data/bids --subjects 01 02 --tasks rest

# Generate report for a specific intermediate result
eeg-generate-reports --bids-root /data/bids \
  --intermediate-results /data/bids/derivatives/nice_preprocessing/intermediate/sub-01_task-rest

# Customize report parameters
eeg-generate-reports --bids-root /data/bids --all \
  --picks eeg \
  --excluded-channels T7 T8
```

**Python API:**
```python
from report_generator import ReportGenerator

# Initialize generator
generator = ReportGenerator(bids_root='/data/bids')

# Generate reports for a specific intermediate result
report_paths = generator.generate_reports(
    intermediate_results_path='/data/bids/derivatives/nice_preprocessing/intermediate/sub-01_task-rest',
    picks_params=['eeg'],
    excluded_channels=['T7', 'T8']
)

print(f"JSON report: {report_paths['json_report']}")
print(f"HTML report: {report_paths['html_report']}")
```

## Workflow Examples

### Workflow 1: Separate Preprocessing and Reporting

```bash
# Step 1: Run preprocessing and save intermediate results
eeg-preprocess-only --bids-root /data/bids --config config.yaml --subjects 01 02

# Step 2: Generate reports (can be done later, on a different machine, etc.)
eeg-generate-reports --bids-root /data/bids --subjects 01 02
```

### Workflow 2: Regenerate Reports with Different Parameters

```bash
# Preprocessing was already done
# Now generate reports with different channel selections

# Default channels
eeg-generate-reports --bids-root /data/bids --all

# Exclude problematic channels
eeg-generate-reports --bids-root /data/bids --all \
  --excluded-channels T7 T8 Fp1 Fp2

# Only EEG channels
eeg-generate-reports --bids-root /data/bids --all --picks eeg
```

### Workflow 3: Process on Server, Report on Laptop

```bash
# On server: Run preprocessing
ssh server "eeg-preprocess-only --bids-root /data/bids --config config.yaml"

# Copy intermediate results to laptop
rsync -av server:/data/bids/derivatives/nice_preprocessing/intermediate/ \
  /local/data/derivatives/nice_preprocessing/intermediate/

# On laptop: Generate reports
eeg-generate-reports --bids-root /local/data --all
```

## Backward Compatibility

The original combined pipeline remains unchanged for backward compatibility:

**Original Files:**
- `src/eeg_preprocessing_pipeline.py` - Original pipeline class
- `src/cli.py` - Original CLI

**Original Console Command:** `eeg-preprocess`

**Usage (unchanged):**
```bash
# This still works exactly as before
eeg-preprocess --bids-root /data/bids --config config.yaml --subjects 01
```

## Directory Structure

```
bids_root/
├── derivatives/
│   └── nice_preprocessing/
│       ├── intermediate/              # NEW: Intermediate results
│       │   ├── sub-01_task-rest/
│       │   │   ├── metadata.json      # Preprocessing metadata
│       │   │   ├── raw.pkl            # Saved Raw object
│       │   │   ├── epochs.pkl         # Saved Epochs object
│       │   │   └── ica.pkl            # Saved ICA object
│       │   └── sub-02_task-rest/
│       │       └── ...
│       ├── reports/                   # JSON and HTML reports
│       │   └── ...
│       ├── clean/                     # Saved clean data (.fif files)
│       │   └── ...
│       ├── preprocessing_results.json  # Preprocessing summary
│       └── report_generation_results.json  # Report generation summary
```

## Configuration File Changes

To use the split workflow, add the `save_intermediate_results` step to your config:

```yaml
pipeline:
  # ... your preprocessing steps ...
  - name: save_intermediate_results  # Add this to save for later reporting
```

If you want to generate reports in the same run (old behavior), add report steps:

```yaml
pipeline:
  # ... your preprocessing steps ...
  - name: generate_json_report
  - name: generate_html_report
    picks: ['eeg']
    excluded_channels: ['T7', 'T8']
```

## Migration Guide

### For Existing Pipelines

**No changes needed!** Your existing configs and scripts will continue to work.

### To Use the Split Workflow

1. Update your config to include `save_intermediate_results` step
2. Remove `generate_json_report` and `generate_html_report` steps (if present)
3. Use `eeg-preprocess-only` instead of `eeg-preprocess`
4. Run `eeg-generate-reports` when you want to create reports

### Example Migration

**Before (config.yaml):**
```yaml
pipeline:
  - name: bandpass_filter
    l_freq: 0.5
    h_freq: 40.0
  - name: ica
    n_components: 20
  - name: generate_json_report
  - name: generate_html_report
```

**After (config.yaml):**
```yaml
pipeline:
  - name: bandpass_filter
    l_freq: 0.5
    h_freq: 40.0
  - name: ica
    n_components: 20
  - name: save_intermediate_results  # Changed from report steps
```

**Command changes:**
```bash
# Before
eeg-preprocess --bids-root /data --config config.yaml

# After (split workflow)
eeg-preprocess-only --bids-root /data --config config.yaml  # Preprocessing
eeg-generate-reports --bids-root /data --all                # Reporting
```

## Benefits of the Split Design

1. **Flexibility**: Generate reports multiple times without reprocessing
2. **Efficiency**: Preprocessing is often computationally expensive; reports are cheap
3. **Experimentation**: Try different report parameters quickly
4. **Debugging**: Inspect intermediate results before report generation
5. **Distributed Computing**: Process on HPC, generate reports locally
6. **Storage**: Delete intermediate results after verification to save space
7. **Reproducibility**: Intermediate results capture exact state for later analysis

## Troubleshooting

### Issue: "Intermediate results not found"

**Solution:** Make sure you've run preprocessing with `save_intermediate_results` step:
```bash
# Check if intermediate results exist
ls -la bids_root/derivatives/nice_preprocessing/intermediate/

# If empty, run preprocessing with save_intermediate_results in config
```

### Issue: "Cannot load pickle file"

**Solution:** Ensure you're using the same Python/MNE version for preprocessing and reporting:
```bash
# Check versions
python -c "import mne; print(f'MNE version: {mne.__version__}')"
```

### Issue: "Memory error when loading intermediate results"

**Solution:** Intermediate results include full MNE objects which can be large:
- Use `--picks eeg` to reduce data size in reports
- Process fewer subjects at once
- Consider using a machine with more RAM

## API Reference

### PreprocessingPipeline

```python
from preprocessing_pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline(
    bids_root: str | Path,
    output_root: str | Path = None,
    config: Dict = None
)

# Run preprocessing
results = pipeline.run_pipeline(
    subjects: List[str] = None,
    sessions: List[str] = None,
    tasks: List[str] = None,
    acquisitions: List[str] = None,
    extension: str = '.vhdr'
)
```

### ReportGenerator

```python
from report_generator import ReportGenerator

generator = ReportGenerator(
    bids_root: str | Path,
    intermediate_results_path: str | Path = None
)

# Load intermediate results
data = generator.load_intermediate_results(
    intermediate_results_path: str | Path = None
)

# Generate JSON report
json_path = generator.generate_json_report(data: Dict)

# Generate HTML report
html_path = generator.generate_html_report(
    data: Dict,
    picks_params: List = None,
    excluded_channels: List = None,
    compare_instances: List = None,
    plot_raw_kwargs: Dict = None,
    plot_ica_kwargs: Dict = None,
    plot_events_kwargs: Dict = None,
    plot_epochs_kwargs: Dict = None,
    plot_evokeds_kwargs: Dict = None,
    n_time_points: int = None
)

# Convenience method: load and generate both reports
report_paths = generator.generate_reports(
    intermediate_results_path: str | Path = None,
    # ... same parameters as generate_html_report ...
)
```

## Support

For issues or questions about the split modules:
1. Check this documentation
2. Review example configs in `configs/`
3. Open an issue on GitHub

## Related Documentation

- Main README: `README.md`
- Configuration examples: `configs/`
- Original implementation: `src/eeg_preprocessing_pipeline.py`
