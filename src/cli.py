#!/usr/bin/env python3
"""
Command-line interface for the EEG preprocessing pipeline.

This module provides CLI commands for running the EEG preprocessing pipeline
from the command line.
"""
import argparse
import json
import yaml
from pathlib import Path
from eeg_preprocessing_pipeline import EEGPreprocessingPipeline


def _parse_args():
    parser = argparse.ArgumentParser(description='Run EEG preprocessing pipeline on one or more subjects.')
    parser.add_argument('--bids-root', required=True, help='Path to BIDS root.')
    parser.add_argument('--output-root', required=False, help='Path to output derivatives root.')
    parser.add_argument(
        '--subjects',
        nargs='+',
        required=True,
        help='Subject ID(s) to process. Provide multiple subject IDs separated by spaces e.g. --subjects 01 02"'
    )
    parser.add_argument('--task', required=False, help='Optional BIDS task label.')
    parser.add_argument('--config', required=False, help='Path to YAML config file with preprocessing parameters.')
    return parser.parse_args()

def main():
    args = _parse_args()
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    pipeline = EEGPreprocessingPipeline(bids_root=args.bids_root, output_root=args.output_root, config=config)
    results = pipeline.run_pipeline(args.subjects, task=args.task)

    # TODO: better logging and result printing
    # print a summary
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
