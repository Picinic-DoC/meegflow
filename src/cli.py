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
        required=False,
        help='Subject ID(s) to process. Provide multiple subject IDs separated by spaces e.g. --subjects 01 02. If not provided, all subjects will be processed.'
    )
    parser.add_argument(
        '--sessions',
        nargs='+',
        required=False,
        help='Session ID(s) to process. Provide multiple session IDs separated by spaces.'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        required=False,
        help='Task(s) to process. Provide multiple task names separated by spaces.'
    )
    parser.add_argument(
        '--acquisitions',
        nargs='+',
        required=False,
        help='Acquisition parameter(s) to process.'
    )
    parser.add_argument(
        '--runs',
        nargs='+',
        required=False,
        help='Run number(s) to process.'
    )
    parser.add_argument(
        '--processings',
        nargs='+',
        required=False,
        help='Processing label(s) to process.'
    )
    parser.add_argument(
        '--recordings',
        nargs='+',
        required=False,
        help='Recording name(s) to process.'
    )
    parser.add_argument(
        '--spaces',
        nargs='+',
        required=False,
        help='Coordinate space(s) to process.'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        required=False,
        help='Split(s) of continuous recording to process.'
    )
    parser.add_argument(
        '--descriptions',
        nargs='+',
        required=False,
        help='Description(s) to process.'
    )
    parser.add_argument('--config', required=False, help='Path to YAML config file with preprocessing parameters.')
    return parser.parse_args()

def main():
    args = _parse_args()
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    pipeline = EEGPreprocessingPipeline(bids_root=args.bids_root, output_root=args.output_root, config=config)
    results = pipeline.run_pipeline(
        subjects=args.subjects,
        sessions=args.sessions,
        tasks=args.tasks,
        acquisitions=args.acquisitions,
        runs=args.runs,
        processings=args.processings,
        recordings=args.recordings,
        spaces=args.spaces,
        splits=args.splits,
        descriptions=args.descriptions,
    )

    # TODO: better logging and result printing
    # print a summary
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
