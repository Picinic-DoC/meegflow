#!/usr/bin/env python3
"""
EEG preprocessing pipeline using MNE-BIDS.

This version updates the CLI and API to accept a list of subjects to process sequentially.
"""
from pathlib import Path
import json
import argparse
from typing import Iterable, Union, Dict, Any
import mne
from mne_bids import BIDSPath, read_raw_bids


class EEGPreprocessingPipeline:
    def __init__(self, bids_root: Union[str, Path], output_root: Union[str, Path] = None, config: Dict[str, Any] = None):
        self.bids_root = Path(bids_root)
        self.output_root = Path(output_root) if output_root is not None else self.bids_root / 'derivatives' / 'nice-preprocessing'
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

    def _process_single_subject(self, subject: str, task: str = None) -> Dict[str, Any]:
        """Process a single subject and return a dictionary of results."""
        results = {'subject': subject}
        bids_path = BIDSPath(root=str(self.bids_root), subject=subject, task=task, datatype='eeg')

        # read_raw_bids will raise a helpful error if files are missing which we capture at call site
        raw = read_raw_bids(bids_path=bids_path)

        # Apply simple preprocessing steps based on config
        l_freq = self.config.get('l_freq', 0.5)
        h_freq = self.config.get('h_freq', 40.0)
        raw.load_data()
        raw.filter(l_freq, h_freq)

        if self.config.get('average_reference', True):
            raw.set_eeg_reference('average', projection=False)

        # ICA step (optional)
        ica = None
        if self.config.get('ica_n_components', 0):
            n_components = self.config.get('ica_n_components')
            ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter='auto')
            picks = mne.pick_types(raw.info, eeg=True, eog=False, meg=False)
            ica.fit(raw, picks=picks)

            # find EOG/ECG components automatically if desired
            if self.config.get('auto_find_eog', True):
                eog_indices, eog_scores = ica.find_bads_eog(raw)
                if eog_indices:
                    ica.exclude.extend(eog_indices)
            if self.config.get('apply_ica', True):
                ica.apply(raw)

        # Epoching -- optional, controlled by config
        events = None
        epochs = None
        if self.config.get('epoching', False):
            events = mne.find_events(raw, shortest_event=1)
            if events is not None and len(events) > 0:
                event_id = self.config.get('event_id', None)
                tmin = self.config.get('tmin', -0.2)
                tmax = self.config.get('tmax', 0.5)
                epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, preload=True)

        # Prepare subject-specific output folder
        subject_out = self.output_root / f'sub-{subject}'
        subject_out.mkdir(parents=True, exist_ok=True)

        # Save epochs if available; otherwise save cleaned raw as fallback
        if epochs is not None:
            safe_task = task or 'unknown'
            epochs_file = subject_out / f'sub-{subject}_task-{safe_task}_clean-epo.fif'
            epochs.save(str(epochs_file), overwrite=True)
            results['epochs_file'] = str(epochs_file)
            results['n_epochs'] = len(epochs)
        else:
            raw_file = subject_out / f'sub-{subject}_raw_clean.fif'
            raw.save(str(raw_file), overwrite=True)
            results['raw_file'] = str(raw_file)

        # Create a simple JSON report
        report = {
            'subject': subject,
            'n_channels': raw.info.get('nchan'),
            'sfreq': raw.info.get('sfreq'),
            'n_times': raw.n_times,
        }
        json_file = subject_out / 'preprocessing_report.json'
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        results['json_report'] = str(json_file)

        return results

    def run_pipeline(self, subjects: Union[str, Iterable[str]], task: str = None) -> Dict[str, Any]:
        """Run the pipeline for a single subject or an iterable of subjects sequentially.

        Parameters
        - subjects: a single subject id (string) or an iterable of subject ids.
        - task: optional BIDS task label used when reading data.

        Returns a dictionary mapping subject -> results dict.
        """
        # Normalize subjects into a list
        if isinstance(subjects, str):
            # allow comma-separated string
            if ',' in subjects:
                subjects_list = [s.strip() for s in subjects.split(',') if s.strip()]
            else:
                subjects_list = [subjects]
        else:
            subjects_list = list(subjects)

        all_results = {}
        for subj in subjects_list:
            try:
                results = self._process_single_subject(subj, task=task)
                all_results[subj] = results
            except Exception as exc:
                # Do not stop the whole batch if one subject fails; capture the error
                all_results[subj] = {'error': str(exc)}
        return all_results


def _parse_args():
    parser = argparse.ArgumentParser(description='Run EEG preprocessing pipeline on one or more subjects.')
    parser.add_argument('--bids-root', required=True, help='Path to BIDS root.')
    parser.add_argument('--output-root', required=False, help='Path to output derivatives root.')
    parser.add_argument(
        '--subjects',
        nargs='+',
        required=True,
        help='Subject ID(s) to process. Provide multiple subject IDs separated by spaces or a single comma-separated string, e.g. --subjects 01 02 or --subjects "01,02"'
    )
    parser.add_argument('--task', required=False, help='Optional BIDS task label.')
    parser.add_argument('--config', required=False, help='Path to JSON config file with preprocessing parameters.')
    return parser.parse_args()

def main():
    args = _parse_args()
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Flatten subjects list (allow comma-separated single string or multiple args)
    subjects_flat = []
    for s in args.subjects:
        if ',' in s:
            parts = [p.strip() for p in s.split(',') if p.strip()]
            subjects_flat.extend(parts)
        else:
            subjects_flat.append(s)

    pipeline = EEGPreprocessingPipeline(bids_root=args.bids_root, output_root=args.output_root, config=config)
    results = pipeline.run_pipeline(subjects_flat, task=args.task)

    # print a summary
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()