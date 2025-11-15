#!/bin/bash
#SBATCH --job-name=eeg_preprocessing
#SBATCH --output=logs/preproc_%A_%a.out
#SBATCH --error=logs/preproc_%A_%a.err
#SBATCH --array=1-20
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute

# SLURM script for batch EEG preprocessing
# 
# Usage:
#   sbatch run_slurm.sh
#
# This script processes multiple subjects in parallel using SLURM job arrays.
# Adjust the --array parameter to match the number of subjects you have.

# Configuration
BIDS_ROOT="/path/to/bids/dataset"
TASK="rest"
SESSION="01"
CONFIG_FILE="config_example.json"  # Optional: leave empty if not using custom config

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (adjust for your cluster)
# Examples:
# module load python/3.9
# module load mne/1.5.0

# Activate virtual environment if using one
# source /path/to/venv/bin/activate

# Get subject ID from SLURM array task ID
# This converts array task ID (1-20) to subject ID with zero-padding (01-20)
SUBJECT=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

echo "=========================================="
echo "Starting preprocessing for subject: $SUBJECT"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Build command
CMD="python eeg_preprocessing_pipeline.py \
    --bids-root $BIDS_ROOT \
    --subject $SUBJECT \
    --task $TASK"

# Add optional parameters
if [ -n "$SESSION" ]; then
    CMD="$CMD --session $SESSION"
fi

if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    CMD="$CMD --config $CONFIG_FILE"
fi

# Run the preprocessing pipeline
echo "Running command: $CMD"
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Preprocessing completed successfully for subject $SUBJECT"
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: Preprocessing failed for subject $SUBJECT"
    echo "=========================================="
    exit 1
fi
