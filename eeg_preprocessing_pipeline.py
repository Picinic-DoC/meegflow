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
        self.config = config or {}
        
        # Map step names to their corresponding methods
        self.step_functions = {
            'load_data': self._step_load_data,
            'bandpass_filter': self._step_bandpass_filter,
            'notch_filter': self._step_notch_filter,
            'reference': self._step_reference,
            'ica': self._step_ica,
            'find_events': self._step_find_events,
            'epoch': self._step_epoch,
            'save_clean_epochs': self._step_save_clean_epochs,
            'generate_json_report': self._step_generate_json_report,
            'generate_html_report': self._step_generate_html_report,
        }
    
    # Auxiliary functions for each preprocessing step

    def _step_load_data(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load raw data into memory."""
        if 'raw' in data:
            data['raw'].load_data()
        return data
    
    def _step_bandpass_filter(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply bandpass filtering."""
        if 'raw' not in data:
            return data

        picks_params = step_config.get('picks', None)
        l_freq = step_config.get('l_freq', 0.5)
        l_freq_order = step_config.get('l_freq_order', 6)
        h_freq = step_config.get('h_freq', 45.0)
        h_freq_order = step_config.get('h_freq_order', 8)
        n_jobs = step_config.get('n_jobs', 1)

        # Get picks if specified
        if picks_params is not None:
            picks = mne.pick_types(data['raw'].info, *picks_params)

        # Apply filtering in 2 steps: high-pass and low-pass
        high_pass_filter_params = dict(
            method='iir',
            l_trans_bandwidth=0.1,
            iir_params=dict(ftype='butter', order=l_freq_order),
            l_freq=l_freq,
            h_freq=None,
            n_jobs=n_jobs
        )
        data['raw'].filter(
            picks=picks,
            **high_pass_filter_params
        )

        low_pass_filter_params = dict(
            method='iir',
            h_trans_bandwidth=0.1,
            iir_params=dict(ftype='butter', order=h_freq_order),
            l_freq=None,
            h_freq=h_freq,
            n_jobs=n_jobs
        )
        data['raw'].filter(
            picks=picks,
            **low_pass_filter_params
        )

        # Store info for reporting
        if 'preprocessing_steps' not in data:
            data['preprocessing_steps'] = []

        data['preprocessing_steps'].extend([
            {
                'step': 'high_pass_filter',
                'picks': picks_params,
                **high_pass_filter_params
            },
            {
                'step': 'low_pass_filter',
                'picks': picks_params,
                **low_pass_filter_params
            }
        ])


        return data

    def _step_notch_filter(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply notch filtering."""
        if 'raw' not in data:
            # TODO: raise exception instead of silent return
            return data

        picks_params = step_config.get('picks', None)
        freqs = step_config.get('freqs', [50.0, 100.0])
        notch_widths = step_config.get('notch_widths', None)
        method = step_config.get('method', 'fft')
        n_jobs = step_config.get('n_jobs', 1)

        # Get picks if specified
        if picks_params is not None:
            picks = mne.pick_types(data['raw'].info, *picks_params)

        data['raw'].notch_filter(
            freqs=freqs,
            method=method,
            notch_widths=notch_widths,
            picks=picks,
            n_jobs=n_jobs
        )

        # Store info for reporting
        if 'preprocessing_steps' not in data:
            data['preprocessing_steps'] = []

        data['preprocessing_steps'].append({
            'step': 'notch_filter',
            'picks': picks_params,
            'freqs': freqs,
            'method': method,
            'notch_widths': notch_widths
        })

        return data

    def _step_reference(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply re-referencing."""

        ref_type = step_config.get('type', 'average')
        instance = step_config.get('instance', 'epochs')
        projection = step_config.get('projection', True)
        apply = step_config.get('apply', True)

        if instance not in data:
            # TODO: raise exception instead of silent return
            return data

        mne.set_eeg_reference(
            inst=data[instance],
            ref_channels=ref_type,
            projection=projection,
            apply=apply,
        )

        if 'preprocessing_steps' not in data:
            data['preprocessing_steps'] = []

        data['preprocessing_steps'].append({
            'step': 'reference',
            'type': ref_type,
            'projection': projection,
            'apply': apply
        })
        
        return data
    
    # TODO: Review ICA
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
            # TODO: raise exception instead of silent return
            return data
        
        shortest_event = step_config.get('shortest_event', 1)
        events = mne.find_events(data['raw'], shortest_event=shortest_event)
        data['events'] = events

        data['preprocessing_steps'].append({
            'step': 'find_events',
            'n_events': len(events)
        })
        
        return data
    
    def _step_epoch(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create epochs from raw data."""
        if data.get('raw', None) is None or data.get('events', None) is None:
            # TODO: raise exception instead of silent return
            return data
        
        event_id = step_config.get('event_id', None)
        tmin = step_config.get('tmin', -0.2)
        tmax = step_config.get('tmax', 0.5)
        baseline = step_config.get('baseline', (None, 0.0))
        reject = step_config.get('reject', None)
        
        data['epochs'] = mne.Epochs(
            data['raw'],
            events=data['events'],
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=reject,
            preload=True
        )

        data['preprocessing_steps'].append({
            'step': 'epoch',
            'event_id': event_id,
            'tmin': tmin,
            'tmax': tmax,
            'baseline': baseline,
            'reject': reject,
            'n_epochs': len(data['epochs'])
        })

        return data

    def _step_save_clean_epochs(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Save clean epochs to a BIDS-derivatives-compatible path."""

        if 'epochs' not in data:
            # TODO: raise exception instead of silent return
            return data

        overwrite = step_config.get('overwrite', True)

        # Derivatives root for this pipeline
        deriv_root = self.bids_root / "derivatives" / "nice_preprocessing" / "epochs"

        bids_path = BIDSPath(
            subject=data['subject'],
            task=data['task'],
            session=data.get('session', None),
            acquisition=data.get('acquisition', None),
            run=data.get('run', None),
            datatype="eeg",
            root=deriv_root,
            suffix="epo",
            extension=".fif",
            processing="clean",
            description="cleaned",
            check=False,
        )

        # Create the directory structure (…/sub-XX[/ses-YY]/eeg)
        bids_path.mkdir(exist_ok=True)

        # Save epochs
        data['epochs'].save(bids_path.fpath, overwrite=overwrite)

        # Store paths & metadata in the data dict
        data['epochs_file'] = str(bids_path.fpath)
        data['n_epochs'] = len(data['epochs'])
        return data
    
    def _step_generate_json_report(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON reports."""
        
        # Derivatives root for this pipeline
        deriv_root = self.bids_root / "derivatives" / "nice_preprocessing" / "reports"

        bids_path = BIDSPath(
            subject=data['subject'],
            task=data['task'],
            session=data.get('session', None),
            acquisition=data.get('acquisition', None),
            run=data.get('run', None),
            datatype="eeg",
            root=deriv_root,
            suffix="report",
            extension=".json",
            processing="clean",
            description="cleaned",
            check=False,
        )

        # Create the directory structure (…/sub-XX[/ses-YY]/eeg)
        bids_path.mkdir(exist_ok=True)
        
        # JSON report
        report = {
            'subject': data['subject'],
            'task': data['task'],
            'session': data.get('session', None),
            'acquisition': data.get('acquisition', None),
            'run': data.get('run', None),
            'preprocessing_steps': data.get('preprocessing_steps', []),
        }
        
        if 'raw' in data:
            raport['raw'] = dict(
                n_channels=data['raw'].info.get('nchan'),
                sfreq=data['raw'].info.get('sfreq'),
                n_times=data['raw'].n_times
            )

        with open(bids_path.fpath, 'w') as f:
            json.dump(report, f, indent=2)
        
        data['json_report'] = str(bids_path.fpath)

    def _step_generate_html_report(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate HTML reports."""
        
        # Derivatives root for this pipeline
        deriv_root = self.bids_root / "derivatives" / "nice_preprocessing" / "reports"

        bids_path = BIDSPath(
            subject=data['subject'],
            task=data['task'],
            session=data.get('session', None),
            acquisition=data.get('acquisition', None),
            run=data.get('run', None),
            datatype="eeg",
            root=deriv_root,
            suffix="report",
            extension=".html",
            processing="clean",
            description="cleaned",
            check=False,
        )

        # Create the directory structure (…/sub-XX[/ses-YY]/eeg)
        bids_path.mkdir(exist_ok=True)
        
        html_report = mne.Report(title=f'Preprocessing Report - Subject {data["subject"]}')

        if 'ica' in data:
            html_report.add_ica(ica=data['ica'], title='ICA Components', inst=data['raw'])
        
        if 'epochs' in data and data['epochs'] is not None:
            html_report.add_epochs(epochs=data['epochs'], title='Clean Epochs')
            evoked = data['epochs'].average()
            html_report.add_evokeds(evokeds=evoked, titles='Average Evoked Response')
        
        html_file = subject_out / 'preprocessing_report.html'
        html_report.save(str(html_file), overwrite=True, open_browser=False)

        data['html_report'] = str(html_file)
        
        return data

    def _process_single_recording(self, bids_path: BIDSPath) -> Dict[str, Any]:
        """Process a single subject using the configured pipeline steps."""
        # Initialize data dictionary
        data = {
            'subject': bids_path.subject,
            'task': bids_path.task,
            'session': bids_path.session,
            'acquisition': bids_path.acquisition,
            'run': bids_path.run,
            'preprocessing_steps': []
        }
        
        # Read BIDS data
        data['raw'] = read_raw_bids(bids_path=bids_path)

        # Get pipeline steps from config
        pipeline_steps = self.config.get('pipeline', [])
        
        # If no pipeline is specified, use default steps
        if not pipeline_steps:
            pipeline_steps = [
                {'name': 'load_data'},
                {'name': 'filter', 'l_freq': 0.5, 'h_freq': 45.0},
                {'name': 'reference', 'type': 'average'},
                {'name': 'save_clean_epochs'},
                {'name': 'generate_json_report'},
                {'name': 'generate_html_report'},
            ]

        # Execute each step in order
        for step in pipeline_steps:
            step_name = step.get('name')
            if step_name not in self.step_functions:
                # TODO: Raiase exception instead of continue with the pipeline. check pipeline steps during initialization
                print(f"Warning: Unknown step '{step_name}', skipping.")
                continue
            
            # Execute the step with its configuration
            step_config = {k: v for k, v in step.items() if k != 'name'}
            data = self.step_functions[step_name](data, step_config)

        # Prepare results
        results = {
            'subject': subject,
            'task': task,
            'session': session,
            'acquisition': acquisition,
            'run': run,
            'raw_file': str(bids_path.fpath)
        }

        # Copy relevant output information to results
        for key in ['raw_file', 'epochs_file', 'json_report', 'html_report', 'n_epochs', 'preprocessing_steps']:
            if key in data:
                results[key] = data[key]

        return results

    def run_pipeline(self, subjects: Iterable[str], task: str = None) -> Dict[str, Any]:
        """Run the pipeline for a single subject or an iterable of subjects sequentially.

        Parameters
        - subjects: an iterable of subject ids.
        - task: optional BIDS task label used when reading data.

        Returns a dictionary mapping subject -> results dict.
        """

        all_results = {}
        for subject in subjects:

            base_path = BIDSPath(
                root=self.bids_root,
                subject=subject,
                task=task,
                datatype='eeg',
                suffix='eeg',
            )

            for raw_path in base_path.match():
                try:
                    results = self._process_single_recording(raw_path)
                    all_results[subject] = results
                except Exception as exc:
                    # Do not stop the whole batch if one subject fails; capture the error
                    all_results[subject] = {'error': str(exc)}

        return all_results


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
    parser.add_argument('--config', required=False, help='Path to JSON config file with preprocessing parameters.')
    return parser.parse_args()

def main():
    args = _parse_args()
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

    pipeline = EEGPreprocessingPipeline(bids_root=args.bids_root, output_root=args.output_root, config=config)
    results = pipeline.run_pipeline(args.subjects, task=args.task)

    # TODO: better logging and result printing
    # print a summary
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()