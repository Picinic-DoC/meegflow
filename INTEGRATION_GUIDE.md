# Integration Guide

This guide explains how to integrate the NICE EEG Preprocessing Pipeline into your project.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data**:
   - Ensure your EEG data is in BIDS format
   - Place it in a directory structure like:
     ```
     bids_dataset/
     ├── sub-01/
     │   └── eeg/
     │       ├── sub-01_task-rest_eeg.edf
     │       └── sub-01_task-rest_eeg.json
     └── dataset_description.json
     ```

3. **Run the pipeline**:
   ```bash
   python eeg_preprocessing_pipeline.py \
       --bids-root /path/to/bids_dataset \
       --subject 01 \
       --task rest
   ```

## Output Structure

After running the pipeline, you'll find three types of outputs:

### 1. Clean Epochs (.fif format)
Location: `derivatives/clean_epochs/`

These are preprocessed epochs ready for analysis. Load them with:
```python
import mne
epochs = mne.read_epochs('derivatives/clean_epochs/sub-01_task-rest_epo.fif')
```

### 2. HTML Reports
Location: `derivatives/html_reports/`

Interactive reports showing:
- Raw data PSD
- ICA components
- Clean epochs visualization
- Average evoked responses

Open in any web browser for review.

### 3. JSON Reports
Location: `derivatives/json_reports/`

Structured metadata for downstream analysis:
```python
import json

with open('derivatives/json_reports/sub-01_task-rest_report.json') as f:
    report = json.load(f)
    
print(f"Epochs kept: {report['preprocessing_info']['epochs']['n_epochs_after_rejection']}")
print(f"Rejection rate: {report['preprocessing_info']['epochs']['rejection_rate']:.2f}%")
```

## Customizing Preprocessing

Create a custom config file based on `config_example.json`:

```json
{
  "l_freq": 1.0,
  "h_freq": 30.0,
  "epochs_tmin": -0.5,
  "epochs_tmax": 1.5,
  "baseline": [-0.2, 0],
  "reject_criteria": {
    "eeg": 1e-04
  },
  "ica_n_components": 15,
  "ica_method": "fastica"
}
```

Use it with:
```bash
python eeg_preprocessing_pipeline.py \
    --bids-root /path/to/bids_dataset \
    --subject 01 \
    --task rest \
    --config my_config.json
```

## Batch Processing

### Local Batch Processing

Edit `run_batch.sh` to set your paths and subjects:
```bash
BIDS_ROOT="/path/to/bids/dataset"
SUBJECTS=("01" "02" "03")
```

Run:
```bash
bash run_batch.sh
```

### SLURM Cluster Processing

Edit `run_slurm.sh` to set your paths:
```bash
BIDS_ROOT="/path/to/bids/dataset"
```

Submit to cluster:
```bash
sbatch run_slurm.sh
```

## Python API

For more control, use the Python API:

```python
from eeg_preprocessing_pipeline import EEGPreprocessingPipeline

# Initialize
pipeline = EEGPreprocessingPipeline(
    bids_root='/path/to/bids/dataset',
    config={
        'l_freq': 1.0,
        'h_freq': 30.0,
        'ica_n_components': 15
    }
)

# Run preprocessing
results = pipeline.run_pipeline(
    subject='01',
    task='rest',
    apply_ica=True
)

# Access results
epochs = results['epochs']
preprocessing_info = results['preprocessing_info']
```

## Pipeline Steps

The pipeline performs these steps automatically:

1. **Read Data**: Loads raw EEG from BIDS dataset
2. **Filter**: Applies bandpass filter (default: 0.5-40 Hz)
3. **Re-reference**: Sets average reference
4. **ICA**: Removes artifacts (optional)
   - Automatic EOG detection
   - Automatic ECG detection
5. **Epoch**: Creates epochs around events
6. **Reject**: Removes bad epochs based on amplitude
7. **Save**: Generates all three output types

## Quality Control

Review the HTML reports to check:
- PSD shows proper filtering
- ICA correctly identified artifacts
- Epochs are clean and well-formed
- Evoked responses show expected patterns

Use the JSON reports to track:
- Number of epochs rejected
- ICA components excluded
- Channel information
- Processing parameters

## Troubleshooting

### No events found
If your data doesn't have event channels, the pipeline creates fixed-length epochs. To use custom events:
```python
events = mne.make_fixed_length_events(raw, duration=1.0)
epochs = pipeline.create_epochs(raw, events=events)
```

### Memory issues
For large datasets, reduce `ica_n_components` or process fewer channels:
```python
config = {'ica_n_components': 10}
```

### ICA not detecting artifacts
Adjust detection threshold:
```python
# In the pipeline, modify these lines:
eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=2.5)  # Lower threshold
```

## Next Steps

After preprocessing:
1. Load epochs for your analysis
2. Extract features (time-frequency, connectivity, etc.)
3. Apply machine learning or statistical tests
4. Use preprocessing info from JSON for quality filtering

## Support

For issues or questions:
- Check the README.md for detailed documentation
- Review example_usage.py for code examples
- Open an issue on GitHub
