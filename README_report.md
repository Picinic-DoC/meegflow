# NICE EEG Report Generation Module

The report generation module creates JSON and HTML reports from preprocessed EEG data. It works independently from the preprocessing module by loading saved intermediate results.

## Features

- **Independent Operation**: Generate reports without re-running preprocessing
- **Multiple Report Formats**: JSON (for programmatic access) and HTML (for visualization)
- **Interactive HTML Reports**: MNE Report-based visualizations with bad channels, preprocessing steps, ICA components, and data quality metrics
- **Batch Processing**: Generate reports for multiple subjects/sessions/tasks
- **Flexible Filtering**: Select which recordings to generate reports for

## Installation

### Option 1: Docker (Recommended)

Build the report generation Docker image:
```bash
docker build -f Dockerfile.report -t nice-report-generator .
```

Or use docker-compose:
```bash
docker-compose build report
```

### Option 2: Local Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Using Docker

Generate reports for all intermediate results:
```bash
docker run --rm \
    -v /path/to/bids:/data \
    nice-report-generator \
    --bids-root /data \
    --all
```

Using docker-compose:
```bash
# Set environment variable
export BIDS_ROOT=/path/to/bids

# Generate reports for all recordings
docker-compose run --rm report --bids-root /data --all

# Generate reports for specific subjects
docker-compose run --rm report --bids-root /data --subjects 01 02
```

### Using Command Line

After installation, use the `eeg-generate-reports` command:

```bash
# Generate reports for all intermediate results
eeg-generate-reports --bids-root /path/to/bids --all

# Generate reports for specific subjects
eeg-generate-reports --bids-root /path/to/bids --subjects 01 02

# Generate reports for specific tasks
eeg-generate-reports --bids-root /path/to/bids --tasks rest

# Generate reports for specific sessions
eeg-generate-reports --bids-root /path/to/bids \
    --subjects 01 \
    --sessions 01 02

# With custom intermediate results location
eeg-generate-reports --bids-root /path/to/bids \
    --intermediate-results /path/to/intermediate \
    --all
```

### Using Python API

```python
from report_generator import ReportGenerator

# Initialize report generator
report_gen = ReportGenerator(bids_root='/path/to/bids')

# Generate reports for all intermediate results
report_gen.generate_all_reports()

# Or generate for specific recording
report_gen.generate_reports(
    subject='01',
    session='01',
    task='rest',
    acquisition=None
)
```

## Prerequisites

Before generating reports, you must:

1. Run the preprocessing module with the `save_intermediate_results` step
2. Ensure intermediate results are saved to the expected location

See [README_preprocessing.md](README_preprocessing.md) for details on running preprocessing.

## Expected Input Structure

The report generator expects intermediate results at:
```
bids_root/derivatives/nice_preprocessing/intermediate/
└── sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}/
    ├── metadata.json          # Required: preprocessing steps and parameters
    ├── raw.pkl                # Optional: raw data object
    ├── epochs.pkl             # Optional: epochs object
    ├── ica.pkl                # Optional: ICA object
    └── events.pkl             # Optional: events array
```

## Output Structure

Reports are saved to:
```
bids_root/derivatives/nice_preprocessing/reports/
└── sub-{subject}/
    └── [ses-{session}/]
        └── eeg/
            ├── sub-{subject}_[ses-{session}_]task-{task}_[acq-{acquisition}_]
            │   processing-clean_desc-cleaned_report.json     # JSON report
            └── sub-{subject}_[ses-{session}_]task-{task}_[acq-{acquisition}_]
                processing-clean_desc-cleaned_report.html     # HTML report
```

## Report Contents

### JSON Report

Contains:
- Subject, session, task, acquisition metadata
- List of all preprocessing steps with parameters
- Data quality metrics (if available)
- Channel information
- Sampling frequency and duration

### HTML Report

Interactive visualizations including:
- **Bad Channels Section**: Topographical plot showing bad channels
- **Preprocessing Steps Table**: Collapsible table with all steps and parameters
- **ICA Components**: If ICA was performed
  - ICA component topoplots
  - EOG/ECG correlation scores
  - Excluded components
- **Raw Data Plots**: Time series of preprocessed data
- **Events Timeline**: If events exist
- **Epochs and Evoked Responses**: If epochs exist
  - Individual epochs
  - Averaged evoked responses
  - Epoch rejection statistics

## Command-Line Arguments

```
usage: eeg-generate-reports [-h] --bids-root BIDS_ROOT
                            [--intermediate-results INTERMEDIATE_RESULTS]
                            [--all] [--subjects SUBJECTS [SUBJECTS ...]]
                            [--sessions SESSIONS [SESSIONS ...]]
                            [--tasks TASKS [TASKS ...]]
                            [--acquisitions ACQUISITIONS [ACQUISITIONS ...]]

Required arguments:
  --bids-root BIDS_ROOT     Path to BIDS root directory

Optional arguments:
  --intermediate-results    Custom path to intermediate results
                           (default: bids_root/derivatives/nice_preprocessing/intermediate)
  --all                    Generate reports for ALL intermediate results
  --subjects                Subject ID(s) to generate reports for
  --sessions                Session ID(s) to generate reports for
  --tasks                   Task(s) to generate reports for
  --acquisitions            Acquisition parameter(s) to generate reports for
```

**Note**: You must specify either `--all` OR at least one filter (subjects/sessions/tasks/acquisitions).

## Examples

### Example 1: Generate All Reports

```bash
eeg-generate-reports --bids-root /data/my_study --all
```

### Example 2: Generate Reports for Specific Subjects

```bash
eeg-generate-reports \
    --bids-root /data/my_study \
    --subjects 01 02 03
```

### Example 3: Generate Reports for Specific Task

```bash
eeg-generate-reports \
    --bids-root /data/my_study \
    --tasks rest
```

### Example 4: Using Docker

```bash
docker run --rm \
    -v /data/my_study:/data \
    nice-report-generator \
    --bids-root /data \
    --subjects 01 02
```

### Example 5: Custom Intermediate Results Location

```bash
eeg-generate-reports \
    --bids-root /data/my_study \
    --intermediate-results /scratch/preprocessed \
    --all
```

## Typical Workflow

### Sequential Workflow (Same Machine)

```bash
# Step 1: Run preprocessing (saves intermediate results)
eeg-preprocess-only \
    --bids-root /data/study \
    --config config_with_save.yaml

# Step 2: Generate reports (can be run immediately or later)
eeg-generate-reports \
    --bids-root /data/study \
    --all
```

### Distributed Workflow (Different Machines)

```bash
# On HPC cluster: Run preprocessing
eeg-preprocess-only \
    --bids-root /mnt/hpc/data/study \
    --config config_with_save.yaml

# Copy intermediate results to local machine
rsync -av user@hpc:/mnt/hpc/data/study/derivatives/nice_preprocessing/intermediate/ \
    /local/data/study/derivatives/nice_preprocessing/intermediate/

# On local machine: Generate reports
eeg-generate-reports \
    --bids-root /local/data/study \
    --all
```

### Docker Workflow

```bash
# Using docker-compose for both steps
export BIDS_ROOT=/data/study
export CONFIG_DIR=$(pwd)/configs

# Step 1: Preprocessing
docker-compose run --rm preprocessing \
    --bids-root /data \
    --config /configs/config_with_save.yaml

# Step 2: Report generation
docker-compose run --rm report \
    --bids-root /data \
    --all
```

## Troubleshooting

### Issue: No intermediate results found

**Solution**: Make sure you ran the preprocessing module with `save_intermediate_results` step in the configuration.

### Issue: Reports are empty or incomplete

**Solution**: Check that the intermediate results directory contains all expected files (metadata.json is required, pickle files are optional but needed for visualizations).

### Issue: Missing visualizations in HTML report

**Solution**: Ensure that the corresponding pickle files (raw.pkl, epochs.pkl, ica.pkl) exist in the intermediate results directory.

### Issue: HTML report won't open

**Solution**: Some browsers block local HTML files. Try:
1. Using a different browser
2. Opening from a local web server (e.g., `python -m http.server`)
3. Checking file permissions

### Issue: Memory errors when loading large datasets

**Solution**: Generate reports for fewer subjects at a time, or increase available memory.

## Security Considerations

**Loading Intermediate Results**: The report module loads pickled data objects. Pickle files from untrusted sources could potentially execute arbitrary code when loaded. Always ensure that:

1. You only load intermediate results from trusted preprocessing runs
2. Intermediate results were created by you or your team
3. You do not load intermediate results from unknown or untrusted sources
4. Intermediate results directories have appropriate file permissions

For production environments:
- Verify the source of intermediate results before loading
- Use separate BIDS roots for different security contexts
- Implement file permission controls on intermediate results directories
- Consider implementing checksums or digital signatures for verification

## Performance Tips

1. **Generate reports in batches**: Process subjects one at a time for large datasets
2. **Delete intermediate results after verification**: Save disk space by removing pickle files once reports are generated and verified
3. **Use filtering**: Generate reports only for subjects/sessions you need
4. **Parallel processing**: Run multiple report generation processes for different subjects simultaneously

## See Also

- [Main README](README.md) - Complete pipeline documentation
- [Preprocessing README](README_preprocessing.md) - Preprocessing module
- [Configuration Examples](configs/) - Example configuration files
- [SPLIT_MODULES_DOCUMENTATION.md](SPLIT_MODULES_DOCUMENTATION.md) - Detailed module documentation
