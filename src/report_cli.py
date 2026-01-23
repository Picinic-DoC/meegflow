#!/usr/bin/env python3
"""
Command-Line Interface for EEG Report Generation.

This module provides a command-line interface for generating reports from
preprocessed EEG data. It loads intermediate results saved by the preprocessing
pipeline and generates JSON and HTML reports.

Main Functions
--------------
report_main() : Entry point for the report generation CLI
    Parses command-line arguments, configures logging, initializes the report
    generator, and creates reports for specified recordings.

Command-Line Usage
------------------
Basic usage (generate reports for a specific intermediate results directory):
    python src/report_cli.py --bids-root /path/to/bids --intermediate-results /path/to/intermediate/results

Generate reports for all intermediate results:
    python src/report_cli.py --bids-root /path/to/bids --all

Filter by subject/session/task:
    python src/report_cli.py --bids-root /path/to/bids --subjects 01 02 --tasks rest

Available Arguments
-------------------
Required:
  --bids-root              Path to BIDS root directory

Specify recordings (choose one):
  --intermediate-results   Path to specific intermediate results directory
  --all                    Generate reports for all intermediate results
  --subjects               Subject ID(s) to generate reports for
  --sessions               Session ID(s) to generate reports for
  --tasks                  Task name(s) to generate reports for
  --acquisitions           Acquisition parameter(s) to generate reports for

Report options:
  --picks                  Channel types to include (e.g., eeg meg)
  --excluded-channels      Channels to exclude from reports

Logging options:
  --log-file              Path to log file (default: console output)
  --log-level             Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)

See README.md for detailed examples and documentation.
"""
import argparse
import json
import pickle
from pathlib import Path
from mne.utils import logger, set_log_file, set_log_level
from report_generator import ReportGenerator


def _parse_args():
    parser = argparse.ArgumentParser(description='Generate reports from preprocessed EEG data.')
    parser.add_argument('--bids-root', required=True, help='Path to BIDS root.')
    parser.add_argument(
        '--intermediate-results',
        required=False,
        help='Path to intermediate results directory for a specific recording.'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate reports for all intermediate results found.'
    )
    parser.add_argument(
        '--subjects',
        nargs='+',
        required=False,
        help='Subject ID(s) to generate reports for.'
    )
    parser.add_argument(
        '--sessions',
        nargs='+',
        required=False,
        help='Session ID(s) to generate reports for.'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        required=False,
        help='Task(s) to generate reports for.'
    )
    parser.add_argument(
        '--acquisitions',
        nargs='+',
        required=False,
        help='Acquisition parameter(s) to generate reports for.'
    )
    parser.add_argument(
        '--picks',
        nargs='+',
        required=False,
        help='Channel types to include in reports (e.g., eeg meg).'
    )
    parser.add_argument(
        '--excluded-channels',
        nargs='+',
        required=False,
        help='Channels to exclude from reports.'
    )
    parser.add_argument('--log-file', required=False, help='Path to log file. If not specified, logs will be printed to console.')
    parser.add_argument('--log-level', required=False, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level (default: INFO).')
    return parser.parse_args()


def _find_intermediate_results(bids_root, subjects=None, sessions=None, tasks=None, acquisitions=None):
    """
    Find all intermediate results directories matching the given criteria.
    
    Returns list of Path objects pointing to intermediate results directories.
    """
    intermediate_root = Path(bids_root) / "derivatives" / "nice_preprocessing" / "intermediate"
    
    if not intermediate_root.exists():
        logger.warning(f"Intermediate results directory not found: {intermediate_root}")
        return []
    
    # Get all subdirectories
    all_dirs = [d for d in intermediate_root.iterdir() if d.is_dir()]
    
    # Filter by criteria
    matching_dirs = []
    for dir_path in all_dirs:
        dir_name = dir_path.name
        
        # Parse directory name: sub-XX_ses-YY_task-ZZ_acq-AA
        parts = dir_name.split('_')
        dir_info = {}
        
        for part in parts:
            if part.startswith('sub-'):
                dir_info['subject'] = part[4:]
            elif part.startswith('ses-'):
                dir_info['session'] = part[4:]
            elif part.startswith('task-'):
                dir_info['task'] = part[5:]
            elif part.startswith('acq-'):
                dir_info['acquisition'] = part[4:]
        
        # Check if matches criteria
        match = True
        if subjects and dir_info.get('subject') not in subjects:
            match = False
        if sessions and dir_info.get('session') not in sessions:
            match = False
        if tasks and dir_info.get('task') not in tasks:
            match = False
        if acquisitions and dir_info.get('acquisition') not in acquisitions:
            match = False
        
        if match:
            matching_dirs.append(dir_path)
    
    return matching_dirs


def report_main():
    args = _parse_args()
    
    # Configure logging
    set_log_level(args.log_level)
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        set_log_file(str(log_path), overwrite=False)
        logger.info(f"Logging to file: {log_path}")
    
    logger.info("Starting EEG report generation")
    logger.info(f"BIDS root: {args.bids_root}")
    
    # Determine which intermediate results to process
    intermediate_paths = []
    
    if args.intermediate_results:
        # Single specific directory
        intermediate_paths = [Path(args.intermediate_results)]
        logger.info(f"Generating report for: {args.intermediate_results}")
    elif args.all or args.subjects or args.sessions or args.tasks or args.acquisitions:
        # Find matching intermediate results
        intermediate_paths = _find_intermediate_results(
            args.bids_root,
            subjects=args.subjects,
            sessions=args.sessions,
            tasks=args.tasks,
            acquisitions=args.acquisitions
        )
        logger.info(f"Found {len(intermediate_paths)} intermediate results to process")
    else:
        logger.error("Must specify --intermediate-results, --all, or filter criteria (--subjects, --tasks, etc.)")
        return
    
    if not intermediate_paths:
        logger.error("No intermediate results found matching criteria")
        return
    
    # Initialize report generator
    generator = ReportGenerator(bids_root=args.bids_root)
    
    # Generate reports for each intermediate result
    results = []
    for i, intermediate_path in enumerate(intermediate_paths, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {i}/{len(intermediate_paths)}: {intermediate_path.name}")
        logger.info(f"{'='*60}")
        
        try:
            report_paths = generator.generate_reports(
                intermediate_results_path=intermediate_path,
                picks_params=args.picks,
                excluded_channels=args.excluded_channels
            )
            
            results.append({
                'intermediate_path': str(intermediate_path),
                'json_report': report_paths['json_report'],
                'html_report': report_paths['html_report'],
                'status': 'success'
            })
            
            logger.info(f"Successfully generated reports:")
            logger.info(f"  JSON: {report_paths['json_report']}")
            logger.info(f"  HTML: {report_paths['html_report']}")
            
        except FileNotFoundError as e:
            logger.error(f"Intermediate results not found for {intermediate_path}: {e}")
            results.append({
                'intermediate_path': str(intermediate_path),
                'status': 'error',
                'error': f'FileNotFoundError: {str(e)}'
            })
        except (IOError, OSError, pickle.UnpicklingError) as e:
            logger.error(f"I/O or unpickling error for {intermediate_path}: {e}")
            results.append({
                'intermediate_path': str(intermediate_path),
                'status': 'error',
                'error': f'{type(e).__name__}: {str(e)}'
            })
        except Exception as e:
            # Re-raise critical errors
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logger.error(f"Unexpected error generating reports for {intermediate_path}: {str(e)}")
            results.append({
                'intermediate_path': str(intermediate_path),
                'status': 'error',
                'error': str(e)
            })
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Report generation completed")
    logger.info("="*60)
    
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = total - successful
    
    logger.info(f"Total: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    # Save results summary
    output_json = Path(args.bids_root) / "derivatives" / "nice_preprocessing" / "report_generation_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults summary written to: {output_json}")


if __name__ == '__main__':
    report_main()
