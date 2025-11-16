# Adaptive Autoreject Integration Guide

This guide explains how to use the newly integrated adaptive autoreject preprocessing steps in the EEG preprocessing pipeline.

## Overview

Four adaptive autoreject functions from `adaptive_reject.py` have been integrated as preprocessing steps:

1. **find_bads_channels_threshold** - Finds bad channels using threshold-based rejection
2. **find_bads_channels_variance** - Finds bad channels using variance-based detection
3. **find_bads_channels_high_frequency** - Finds bad channels using high-frequency variance
4. **find_bads_epochs_threshold** - Finds and removes bad epochs using threshold-based rejection

## When to Use Each Step

### Channel Detection Steps

These steps should typically be run **after epoching** and **before epoch rejection**:

1. **find_bads_channels_threshold**: Use when you want to identify channels that consistently exceed amplitude thresholds across many epochs.

2. **find_bads_channels_variance**: Use to detect channels with abnormally high or low variance (e.g., disconnected electrodes or flatlined channels).

3. **find_bads_channels_high_frequency**: Use to identify channels with excessive high-frequency noise (e.g., muscle artifacts).

### Epoch Rejection Step

**find_bads_epochs_threshold** should be run **after** channel detection steps to remove epochs with too many bad channels.

## Basic Usage

### Command Line

```bash
python src/eeg_preprocessing_pipeline.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 03 \
    --task rest \
    --config configs/config_with_adaptive_reject.json
```

### Python API

```python
from eeg_preprocessing_pipeline import EEGPreprocessingPipeline
import json

# Load configuration
with open('configs/config_with_adaptive_reject.json', 'r') as f:
    config = json.load(f)

# Initialize pipeline
pipeline = EEGPreprocessingPipeline(
    bids_root='/path/to/bids/dataset',
    config=config
)

# Run preprocessing
results = pipeline.run_pipeline(
    subjects=['01', '02', '03'],
    task='rest'
)
```

## Configuration Parameters

### find_bads_channels_threshold

```json
{
  "name": "find_bads_channels_threshold",
  "picks": null,  // Optional: Channel indices (default: EEG channels)
  "reject": {
    "eeg": 1.5e-04  // Rejection threshold in V (150 µV)
  },
  "n_epochs_bad_ch": 0.5  // Fraction of epochs (0.5 = 50%)
}
```

- **picks**: Which channels to check (default: all EEG channels)
- **reject**: Amplitude thresholds by channel type
- **n_epochs_bad_ch**: A channel is marked bad if it exceeds threshold in this many epochs (can be fraction 0-1 or absolute number)

### find_bads_channels_variance

```json
{
  "name": "find_bads_channels_variance",
  "instance": "epochs",  // Can be "raw" or "epochs"
  "picks": null,  // Optional: Channel indices
  "zscore_thresh": 4,  // Z-score threshold for outliers
  "max_iter": 2  // Maximum iterations for outlier removal
}
```

- **instance**: Apply to 'raw' data or 'epochs' (default: 'epochs')
- **picks**: Which channels to check (default: all EEG channels)
- **zscore_thresh**: How many standard deviations from mean to consider outlier
- **max_iter**: Iteratively remove outliers this many times

### find_bads_channels_high_frequency

```json
{
  "name": "find_bads_channels_high_frequency",
  "instance": "epochs",  // Can be "raw" or "epochs"
  "picks": null,  // Optional: Channel indices
  "zscore_thresh": 4,  // Z-score threshold for outliers
  "max_iter": 2  // Maximum iterations
}
```

- Parameters are identical to `find_bads_channels_variance`
- Internally applies high-pass filter at 25 Hz before variance calculation

### find_bads_epochs_threshold

```json
{
  "name": "find_bads_epochs_threshold",
  "picks": null,  // Optional: Channel indices
  "reject": {
    "eeg": 1.5e-04  // Rejection threshold in V
  },
  "n_channels_bad_epoch": 0.1  // Fraction of channels (0.1 = 10%)
}
```

- **picks**: Which channels to check (default: all EEG channels)
- **reject**: Amplitude thresholds by channel type
- **n_channels_bad_epoch**: An epoch is rejected if this many channels are bad (can be fraction 0-1 or absolute number)

## Recommended Pipeline Order

```json
{
  "pipeline": [
    {"name": "load_data"},
    {"name": "bandpass_filter", "l_freq": 0.5, "h_freq": 40.0},
    {"name": "reference", "type": "average"},
    {"name": "ica", "n_components": 20, "find_eog": true, "apply": true},
    {"name": "find_events"},
    {"name": "epoch", "tmin": -0.2, "tmax": 0.8, "baseline": [null, 0]},
    
    // Adaptive autoreject steps
    {"name": "find_bads_channels_threshold", "reject": {"eeg": 1.5e-04}, "n_epochs_bad_ch": 0.5},
    {"name": "find_bads_channels_variance", "zscore_thresh": 4},
    {"name": "find_bads_channels_high_frequency", "zscore_thresh": 4},
    {"name": "find_bads_epochs_threshold", "reject": {"eeg": 1.5e-04}, "n_channels_bad_epoch": 0.1},
    
    {"name": "save_clean_epochs"},
    {"name": "generate_json_report"},
    {"name": "generate_html_report"}
  ]
}
```

## What Happens Under the Hood

### Channel Detection Steps

1. The function identifies bad channels based on the specified criteria
2. Bad channels are added to `epochs.info['bads']` (or `raw.info['bads']`)
3. MNE will automatically interpolate these channels when needed
4. The preprocessing report includes the list of bad channels found

### Epoch Rejection Step

1. The function identifies epochs with too many bad channels
2. These epochs are dropped from the Epochs object using `epochs.drop()`
3. The preprocessing report includes the list of rejected epoch indices
4. Only clean epochs are saved to the output file

## Accessing Results

The preprocessing report (JSON format) includes detailed information about each step:

```python
import json

# Load the report
with open('path/to/report.json', 'r') as f:
    report = json.load(f)

# Access adaptive reject results
for step in report['preprocessing_steps']:
    if step['step'] == 'find_bads_channels_threshold':
        print(f"Bad channels found: {step['bad_channels']}")
        print(f"Number of bad channels: {step['n_bad_channels']}")
    
    if step['step'] == 'find_bads_epochs_threshold':
        print(f"Bad epochs removed: {step['bad_epochs']}")
        print(f"Epochs remaining: {step['n_epochs_remaining']}")
```

## Tips and Best Practices

1. **Start conservative**: Use higher thresholds initially and adjust based on your data quality

2. **Order matters**: Run channel detection before epoch rejection to avoid rejecting epochs due to bad channels

3. **Multiple passes**: You can include channel detection steps multiple times if needed

4. **Raw vs Epochs**: 
   - Use `instance: "raw"` for channel variance/high-frequency detection on continuous data
   - Use `instance: "epochs"` (default) for epoched data

5. **Check the reports**: Always review the JSON/HTML reports to see how many channels/epochs were rejected

6. **Adjust parameters**: Different datasets may require different thresholds - tune based on your data

## Example: Different Use Cases

### Minimal Artifact Rejection
Only remove very bad channels and epochs:

```json
{"name": "find_bads_channels_threshold", "reject": {"eeg": 2.0e-04}, "n_epochs_bad_ch": 0.7},
{"name": "find_bads_epochs_threshold", "reject": {"eeg": 2.0e-04}, "n_channels_bad_epoch": 0.2}
```

### Aggressive Cleaning
More strict rejection for high-quality data:

```json
{"name": "find_bads_channels_threshold", "reject": {"eeg": 1.0e-04}, "n_epochs_bad_ch": 0.3},
{"name": "find_bads_channels_variance", "zscore_thresh": 3},
{"name": "find_bads_channels_high_frequency", "zscore_thresh": 3},
{"name": "find_bads_epochs_threshold", "reject": {"eeg": 1.0e-04}, "n_channels_bad_epoch": 0.05}
```

### Focus on Noise
Only use high-frequency and variance detection:

```json
{"name": "find_bads_channels_variance", "zscore_thresh": 4},
{"name": "find_bads_channels_high_frequency", "zscore_thresh": 4}
```

## Troubleshooting

### Too many channels rejected
- Increase `n_epochs_bad_ch` threshold
- Increase `zscore_thresh` values
- Check your data quality and preprocessing steps before adaptive reject

### Too many epochs rejected
- Increase `n_channels_bad_epoch` threshold
- Adjust `reject` thresholds to be less strict
- Consider interpolating bad channels instead of rejecting epochs

### No channels/epochs rejected
- Decrease thresholds to be more strict
- Check that you're passing the correct data instance ('raw' vs 'epochs')
- Verify data has been properly loaded and preprocessed

## References

Original adaptive reject implementation by Federico Raimondo (2017)
- Based on principles from autoreject: Jas, M., et al. (2017). Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage.
