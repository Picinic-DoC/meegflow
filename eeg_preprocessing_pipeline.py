#!/usr/bin/env python3
"""
EEG preprocessing pipeline using MNE-BIDS.

This version is modular with separate functions for each preprocessing step.
The pipeline is configuration-driven - you specify steps, their order, and parameters.
"""
from pathlib import Path
import json
import argparse
from typing import Iterable, Union, Dict, Any, List, Callable
import mne
from mne_bids import BIDSPath, read_raw_bids


class EEGPreprocessingPipeline:
    def __init__(self, bids_root: Union[str, Path], output_root: Union[str, Path] = None, config: Dict[str, Any] = None):
        self.bids_root = Path(bids_root)
        self.output_root = Path(output_root) if output_root is not None else self.bids_root / 'derivatives' / 'nice-preprocessing'
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        # Map step names to their corresponding methods
        self.step_functions = {
            'load_data': self._step_load_data,
            'filter': self._step_filter,
            'reference': self._step_reference,
            'ica': self._step_ica,
            'find_events': self._step_find_events,
            'epoch': self._step_epoch,
            'save_clean_epochs': self._step_save_clean_epochs,
            'save_clean_raw': self._step_save_clean_raw,
            'generate_report': self._step_generate_report,
        }
    
    # Auxiliary functions for each preprocessing step
    
    def _step_load_data(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load raw data into memory."""
        if 'raw' in data:
            data['raw'].load_data()
        return data
    
    def _step_filter(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply bandpass filtering."""
        if 'raw' not in data:
            return data
        
        l_freq = step_config.get('l_freq', 0.5)
        h_freq = step_config.get('h_freq', 40.0)
        data['raw'].filter(l_freq, h_freq)
        
        # Store info for reporting
        if 'preprocessing_steps' not in data:
            data['preprocessing_steps'] = []
        data['preprocessing_steps'].append({
            'step': 'filter',
            'l_freq': l_freq,
            'h_freq': h_freq
        })
        
        return data
    
    def _step_reference(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply re-referencing."""
        if 'raw' not in data:
            return data
        
        ref_type = step_config.get('type', 'average')
        projection = step_config.get('projection', False)
        
        if ref_type == 'average':
            data['raw'].set_eeg_reference('average', projection=projection)
        
        if 'preprocessing_steps' not in data:
            data['preprocessing_steps'] = []
        data['preprocessing_steps'].append({
            'step': 'reference',
            'type': ref_type,
            'projection': projection
        })
        
        return data
    
    def _step_ica(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ICA for artifact removal."""
        if 'raw' not in data:
            return data
        
        n_components = step_config.get('n_components', 20)
        random_state = step_config.get('random_state', 97)
        method = step_config.get('method', 'fastica')
        
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=random_state,
            method=method,
            max_iter='auto'
        )
        picks = mne.pick_types(data['raw'].info, eeg=True, eog=False, meg=False)
        ica.fit(data['raw'], picks=picks)
        
        # Automatically find and exclude artifacts
        excluded_components = []
        if step_config.get('find_eog', True):
            try:
                eog_indices, eog_scores = ica.find_bads_eog(data['raw'])
                if eog_indices:
                    ica.exclude.extend(eog_indices)
                    excluded_components.extend(['eog'] * len(eog_indices))
            except:
                pass  # No EOG channels available
        
        if step_config.get('find_ecg', False):
            try:
                ecg_indices, ecg_scores = ica.find_bads_ecg(data['raw'])
                if ecg_indices:
                    ica.exclude.extend(ecg_indices)
                    excluded_components.extend(['ecg'] * len(ecg_indices))
            except:
                pass  # No ECG channels available
        
        # Apply ICA to remove artifacts
        if step_config.get('apply', True):
            ica.apply(data['raw'])
        
        data['ica'] = ica
        
        if 'preprocessing_steps' not in data:
            data['preprocessing_steps'] = []
        data['preprocessing_steps'].append({
            'step': 'ica',
            'n_components': n_components,
            'method': method,
            'excluded_components': len(ica.exclude),
            'component_types': excluded_components
        })
        
        return data
    
    def _step_find_events(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Find events in the data."""
        if 'raw' not in data:
            return data
        
        shortest_event = step_config.get('shortest_event', 1)
        try:
            events = mne.find_events(data['raw'], shortest_event=shortest_event)
            data['events'] = events
            
            if 'preprocessing_steps' not in data:
                data['preprocessing_steps'] = []
            data['preprocessing_steps'].append({
                'step': 'find_events',
                'n_events': len(events)
            })
        except:
            data['events'] = None
        
        return data
    
    def _step_epoch(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create epochs from raw data."""
        if 'raw' not in data or 'events' not in data:
            return data
        
        if data['events'] is None or len(data['events']) == 0:
            return data
        
        event_id = step_config.get('event_id', None)
        tmin = step_config.get('tmin', -0.2)
        tmax = step_config.get('tmax', 0.5)
        baseline = step_config.get('baseline', None)
        reject = step_config.get('reject', None)
        
        epochs = mne.Epochs(
            data['raw'],
            events=data['events'],
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=reject,
            preload=True
        )
        
        data['epochs'] = epochs
        
        if 'preprocessing_steps' not in data:
            data['preprocessing_steps'] = []
        data['preprocessing_steps'].append({
            'step': 'epoch',
            'tmin': tmin,
            'tmax': tmax,
            'n_epochs': len(epochs)
        })
        
        return data
    
    def _step_save_clean_epochs(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Save clean epochs to .fif file."""
        if 'epochs' not in data or data['epochs'] is None:
            return data
        
        subject = data.get('subject', 'unknown')
        task = data.get('task', 'unknown')
        
        subject_out = self.output_root / f'sub-{subject}' / 'clean_epochs'
        subject_out.mkdir(parents=True, exist_ok=True)
        
        epochs_file = subject_out / f'sub-{subject}_task-{task}_clean-epo.fif'
        data['epochs'].save(str(epochs_file), overwrite=True)
        
        data['epochs_file'] = str(epochs_file)
        data['n_epochs'] = len(data['epochs'])
        
        return data
    
    def _step_save_clean_raw(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Save clean raw data to .fif file."""
        if 'raw' not in data:
            return data
        
        subject = data.get('subject', 'unknown')
        
        subject_out = self.output_root / f'sub-{subject}' / 'clean_raw'
        subject_out.mkdir(parents=True, exist_ok=True)
        
        raw_file = subject_out / f'sub-{subject}_raw_clean.fif'
        data['raw'].save(str(raw_file), overwrite=True)
        
        data['raw_file'] = str(raw_file)
        
        return data
    
    def _step_generate_report(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON and HTML reports."""
        subject = data.get('subject', 'unknown')
        
        subject_out = self.output_root / f'sub-{subject}' / 'reports'
        subject_out.mkdir(parents=True, exist_ok=True)
        
        # JSON report
        report = {
            'subject': subject,
            'task': data.get('task', None),
            'preprocessing_steps': data.get('preprocessing_steps', []),
        }
        
        if 'raw' in data:
            report['n_channels'] = data['raw'].info.get('nchan')
            report['sfreq'] = data['raw'].info.get('sfreq')
            report['n_times'] = data['raw'].n_times
        
        if 'epochs' in data and data['epochs'] is not None:
            report['n_epochs'] = len(data['epochs'])
        
        json_file = subject_out / 'preprocessing_report.json'
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        data['json_report'] = str(json_file)
        
        # HTML report (optional, requires more dependencies)
        if step_config.get('generate_html', False):
            try:
                html_report = mne.Report(title=f'Preprocessing Report - Subject {subject}')
                
                if 'raw' in data:
                    html_report.add_raw(raw=data['raw'], title='Preprocessed Raw Data', psd=True)
                
                if 'ica' in data:
                    html_report.add_ica(ica=data['ica'], title='ICA Components', inst=data['raw'])
                
                if 'epochs' in data and data['epochs'] is not None:
                    html_report.add_epochs(epochs=data['epochs'], title='Clean Epochs')
                    evoked = data['epochs'].average()
                    html_report.add_evokeds(evokeds=evoked, titles='Average Evoked Response')
                
                html_file = subject_out / 'preprocessing_report.html'
                html_report.save(str(html_file), overwrite=True, open_browser=False)
                data['html_report'] = str(html_file)
            except Exception as e:
                print(f"Warning: Could not generate HTML report: {e}")
        
        return data

    def _process_single_subject(self, subject: str, task: str = None) -> Dict[str, Any]:
        """Process a single subject using the configured pipeline steps."""
        # Initialize data dictionary
        data = {
            'subject': subject,
            'task': task,
        }
        
        # Read BIDS data
        bids_path = BIDSPath(root=str(self.bids_root), subject=subject, task=task, datatype='eeg')
        data['raw'] = read_raw_bids(bids_path=bids_path)
        
        # Get pipeline steps from config
        pipeline_steps = self.config.get('pipeline', [])
        
        # If no pipeline is specified, use default steps
        if not pipeline_steps:
            pipeline_steps = [
                {'name': 'load_data'},
                {'name': 'filter', 'l_freq': 0.5, 'h_freq': 40.0},
                {'name': 'reference', 'type': 'average'},
                {'name': 'save_clean_raw'},
                {'name': 'generate_report'},
            ]
        
        # Execute each step in order
        for step in pipeline_steps:
            step_name = step.get('name')
            if step_name not in self.step_functions:
                print(f"Warning: Unknown step '{step_name}', skipping.")
                continue
            
            # Execute the step with its configuration
            step_config = {k: v for k, v in step.items() if k != 'name'}
            data = self.step_functions[step_name](data, step_config)
        
        # Prepare results
        results = {
            'subject': subject,
            'task': task,
        }
        
        # Copy relevant output information to results
        for key in ['epochs_file', 'raw_file', 'json_report', 'html_report', 'n_epochs', 'preprocessing_steps']:
            if key in data:
                results[key] = data[key]
        
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