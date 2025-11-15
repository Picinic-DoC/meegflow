#!/bin/bash

# Batch processing script for EEG preprocessing
# 
# Usage:
#   bash run_batch.sh
#
# This script processes multiple subjects sequentially on a local machine.

# Configuration
BIDS_ROOT="/path/to/bids/dataset"
TASK="rest"
SESSION="01"
CONFIG_FILE=""  # Optional: path to config file

# List of subjects to process
SUBJECTS=("01" "02" "03" "04" "05")

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Starting batch EEG preprocessing"
echo "Number of subjects: ${#SUBJECTS[@]}"
echo "=========================================="

# Process each subject
for subject in "${SUBJECTS[@]}"; do
    echo ""
    echo "Processing subject: $subject"
    echo "------------------------------------------"
    
    # Build command
    CMD="python eeg_preprocessing_pipeline.py \
        --bids-root $BIDS_ROOT \
        --subject $subject \
        --task $TASK"
    
    # Add optional parameters
    if [ -n "$SESSION" ]; then
        CMD="$CMD --session $SESSION"
    fi
    
    if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
        CMD="$CMD --config $CONFIG_FILE"
    fi
    
    # Run preprocessing and log output
    LOG_FILE="logs/subject_${subject}.log"
    echo "Running: $CMD"
    eval $CMD 2>&1 | tee "$LOG_FILE"
    
    # Check exit status
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Subject $subject completed successfully"
    else
        echo "✗ ERROR: Subject $subject failed"
        echo "See log file: $LOG_FILE"
        # Uncomment the next line to stop on first error
        # exit 1
    fi
    
    echo "------------------------------------------"
done

echo ""
echo "=========================================="
echo "Batch processing completed!"
echo "Logs saved to: logs/"
echo "=========================================="
