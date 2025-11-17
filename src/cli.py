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
from mne.utils import logger, set_log_file, set_log_level
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
    parser.add_argument('--log-file', required=False, help='Path to log file. If not specified, logs will be printed to console.')
    parser.add_argument('--log-level', required=False, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level (default: INFO).')
    return parser.parse_args()

def main():
    args = _parse_args()
    
    # Configure logging
    set_log_level(args.log_level)
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        set_log_file(str(log_path), overwrite=False)
        logger.info(f"Logging to file: {log_path}")
    
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    logger.info("Starting EEG preprocessing pipeline")
    logger.info(f"BIDS root: {args.bids_root}")
    if args.output_root:
        logger.info(f"Output root: {args.output_root}")
    
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

    # Log summary of results
    logger.info("=" * 60)
    logger.info("Pipeline execution completed")
    logger.info("=" * 60)
    
    total_recordings = sum(len(v) for v in results.values())
    total_errors = sum(1 for subj_results in results.values() for r in subj_results if 'error' in r)
    total_success = total_recordings - total_errors
    
    logger.info(f"Total recordings processed: {total_recordings}")
    logger.info(f"Successful: {total_success}")
    logger.info(f"Failed: {total_errors}")
    
    # Log detailed results in JSON format
    logger.info("\nDetailed results:")
    for subject, subject_results in results.items():
        logger.info(f"\nSubject {subject}:")
        for result in subject_results:
            if 'error' in result:
                logger.error(f"  - Error: {result['error']}")
            else:
                task = result.get('task', 'N/A')
                session = result.get('session', 'N/A')
                n_epochs = result.get('n_epochs', 'N/A')
                logger.info(f"  - Task: {task}, Session: {session}, Epochs: {n_epochs}")
    
    # Also write results to JSON file for easy processing
    output_json = Path(args.bids_root) / "derivatives" / "nice_preprocessing" / "pipeline_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults written to: {output_json}")


if __name__ == '__main__':
    main()
