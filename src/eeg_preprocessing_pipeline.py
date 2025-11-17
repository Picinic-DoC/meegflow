#!/usr/bin/env python3
"""
EEG preprocessing pipeline using MNE-BIDS.

This version is modular with separate functions for each preprocessing step.
The pipeline is configuration-driven - you specify steps, their order, and parameters.
"""
from pathlib import Path
from typing import Iterable, Union, Dict, Any, List
import json
import mne
from mne.utils import logger
import numpy as np
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import adaptive_reject


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
            'find_bads_channels_threshold': self._step_find_bads_channels_threshold,
            'find_bads_channels_variance': self._step_find_bads_channels_variance,
            'find_bads_channels_high_frequency': self._step_find_bads_channels_high_frequency,
            'find_bads_epochs_threshold': self._step_find_bads_epochs_threshold,
            'save_clean_epochs': self._step_save_clean_epochs,
            'generate_json_report': self._step_generate_json_report,
            'generate_html_report': self._step_generate_html_report,
        }

        # Validate pipeline steps if provided in config
        pipeline_cfg = self.config.get('pipeline', [])
        unknown = [s.get('name') for s in pipeline_cfg if s.get('name') not in self.step_functions]
        if unknown:
            raise ValueError(f"Unknown pipeline steps in config: {unknown}")

    def _get_pipeline_steps(self) -> List[Dict[str, Any]]:
        """Retrieve the list of pipeline steps from the configuration."""
        pipeline_steps = self.config.get('pipeline', [])

        if not pipeline_steps:
            pipeline_steps = [
                {'name': 'load_data'},
                {'name': 'bandpass_filter', 'l_freq': 0.5, 'h_freq': 45.0},
                {'name': 'reference', 'type': 'average'},
                {'name': 'save_clean_epochs'},
                {'name': 'generate_json_report'},
                {'name': 'generate_html_report'},
            ]
    
        return pipeline_steps

    def _get_picks(self, info: mne.Info, picks_params: Any) -> List[int]:
        # Compute picks if provided
        if isinstance(picks_params, (list, tuple)):
            return  mne.pick_types(info, *picks_params)
        return None

    # Auxiliary functions for each preprocessing step

    def _step_load_data(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load raw data into memory."""
        if 'raw' in data:
            data['raw'].load_data()
        return data

    def _step_bandpass_filter(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply bandpass filtering."""
        if 'raw' not in data:
            raise ValueError("bandpass_filter requires 'raw' in data")

        picks_params = step_config.get('picks', None)
        l_freq = step_config.get('l_freq', 0.5)
        l_freq_order = step_config.get('l_freq_order', 6)
        h_freq = step_config.get('h_freq', 45.0)
        h_freq_order = step_config.get('h_freq_order', 8)
        n_jobs = step_config.get('n_jobs', 1)

        # Compute picks if provided, otherwise None (all channels)
        picks = self._get_picks(data['raw'].info, picks_params)

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
        data['preprocessing_steps'].extend([
            {
                'step': 'high_pass_filter',
                'picks': picks_params,
                'params': high_pass_filter_params
            },
            {
                'step': 'low_pass_filter',
                'picks': picks_params,
                'params': low_pass_filter_params
            }
        ])

        return data

    def _step_notch_filter(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply notch filtering."""
        if 'raw' not in data:
            raise ValueError("notch_filter requires 'raw' in data")

        picks_params = step_config.get('picks', None)
        freqs = step_config.get('freqs', [50.0, 100.0])
        notch_widths = step_config.get('notch_widths', None)
        method = step_config.get('method', 'fft')
        n_jobs = step_config.get('n_jobs', 1)

        # Compute picks if provided
        picks = self._get_picks(data['raw'].info, picks_params)

        data['raw'].notch_filter(
            freqs=freqs,
            method=method,
            notch_widths=notch_widths,
            picks=picks,
            n_jobs=n_jobs
        )

        # Store info for reporting
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

        ref_channels = step_config.get('ref_channels', 'average')
        instance = step_config.get('instance', 'epochs')

        if instance not in data:
            raise ValueError(f"reference step requires '{instance}' to be present in data (either 'raw' or 'epochs')")

        mne.set_eeg_reference(
            inst=data[instance],
            ref_channels=ref_channels,
        )

        data['preprocessing_steps'].append({
            'step': 'reference',
            'ref_channels': ref_channels
        })

        return data

    # Improved ICA with safer channel selection and explicit exception handling
    def _step_ica(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ICA for artifact removal."""
        if 'raw' not in data:
            raise ValueError("ica step requires 'raw' in data")

        n_components = step_config.get('n_components', 20)
        random_state = step_config.get('random_state', 97)
        method = step_config.get('method', 'fastica')

        ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=random_state,
            method=method,
            max_iter='auto'
        )
        # Restrict picks to EEG channels only, if present
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
            except Exception:
                # no EOG channels or detection failed
                pass

        if step_config.get('find_ecg', False):
            try:
                ecg_indices, ecg_scores = ica.find_bads_ecg(data['raw'])
                if ecg_indices:
                    ica.exclude.extend(ecg_indices)
                    excluded_components.extend(['ecg'] * len(ecg_indices))
            except Exception:
                pass

        # Apply ICA to remove artifacts if requested
        if step_config.get('apply', True):
            ica.apply(data['raw'])

        data['ica'] = ica

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
            raise ValueError("find_events requires 'raw' in data")

        shortest_event = step_config.get('shortest_event', 1)
        try:
            # First attempt: use stim channel(s)
            events = mne.find_events(data['raw'], shortest_event=shortest_event)
        except ValueError as e:
            # If no stim channel found → fallback to annotations
            if "No stim channels found" in str(e):
                events, event_id = mne.events_from_annotations(data['raw'], event_id='auto')
            else:
                # re-raise if it's some other error
                raise
        data['events'] = events

        data['preprocessing_steps'].append({
            'step': 'find_events',
            'n_events': len(events)
        })

        return data

    def _step_epoch(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create epochs from raw data."""
        if data.get('raw', None) is None or data.get('events', None) is None:
            raise ValueError("epoch step requires both 'raw' and 'events' in data")

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

    def _step_find_bads_channels_threshold(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Find bad channels using threshold-based rejection."""
        if 'epochs' not in data:
            raise ValueError("find_bads_channels_threshold requires 'epochs' in data")

        picks_params = step_config.get('picks', None)
        reject = step_config.get('reject', {'eeg': 150e-6})
        n_epochs_bad_ch = step_config.get('n_epochs_bad_ch', 0.5)
        apply_on = step_config.get('apply_on', ['epochs'])

        if not isinstance(apply_on, list):
            apply_on = [apply_on]

        if any(inst not in data for inst in apply_on):
            raise ValueError(f"find_bads_channels_threshold requires all instances of apply_on ({apply_on}) to be present in data")

        picks = self._get_picks(data['raw'].info, picks_params)

        bad_chs = adaptive_reject.find_bads_channels_threshold(
            data['epochs'], picks, reject, n_epochs_bad_ch
        )

        if bad_chs:
            for instance_name in apply_on:
                data[instance_name].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_name].info['bads']])

        data['preprocessing_steps'].append({
            'step': 'find_bads_channels_threshold',
            'picks': picks_params,
            'apply_on': apply_on,
            'reject': reject,
            'n_epochs_bad_ch': n_epochs_bad_ch,
            'bad_channels': bad_chs,
            'n_bad_channels': len(bad_chs)
        })

        return data

    def _step_find_bads_channels_variance(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Find bad channels using variance-based detection."""
        # Check which instance to use
        instance_name = step_config.get('instance', 'epochs')
        if instance_name not in data:
            raise ValueError(f"find_bads_channels_variance requires '{instance_name}' in data")

        picks_params = step_config.get('picks', None)
        inst = data[instance_name]
        zscore_thresh = step_config.get('zscore_thresh', 4)
        max_iter = step_config.get('max_iter', 2)
        apply_on = step_config.get('apply_on', [instance_name])

        if not isinstance(apply_on, list):
            apply_on = [apply_on]

        if any(inst not in data for inst in apply_on):
            raise ValueError(f"find_bads_channels_threshold requires all instances of apply_on ({apply_on}) to be present in data")

        picks = self._get_picks(data['raw'].info, picks_params)

        bad_chs = adaptive_reject.find_bads_channels_variance(
            inst, picks, zscore_thresh, max_iter
        )

        # Mark channels as bad
        if bad_chs:
            for instance_name in apply_on:
                data[instance_name].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_name].info['bads']])

        data['preprocessing_steps'].append({
            'step': 'find_bads_channels_variance',
            'instance': instance_name,
            'picks': picks_params,
            'apply_on': apply_on,
            'zscore_thresh': zscore_thresh,
            'max_iter': max_iter,
            'bad_channels': bad_chs,
            'n_bad_channels': len(bad_chs)
        })

        return data

    def _step_find_bads_channels_high_frequency(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Find bad channels using high-frequency variance."""
        # Check which instance to use
        instance_name = step_config.get('instance', 'epochs')
        if instance_name not in data:
            raise ValueError(f"find_bads_channels_high_frequency requires '{instance_name}' in data")

        picks_params = step_config.get('picks', None)
        inst = data[instance_name]
        zscore_thresh = step_config.get('zscore_thresh', 4)
        max_iter = step_config.get('max_iter', 2)
        apply_on = step_config.get('apply_on', [instance_name])

        if not isinstance(apply_on, list):
            apply_on = [apply_on]
        
        if any(inst not in data for inst in apply_on):
            raise ValueError(f"find_bads_channels_threshold requires all instances of apply_on ({apply_on}) to be present in data")

        picks = self._get_picks(data['raw'].info, picks_params)

        bad_chs = adaptive_reject.find_bads_channels_high_frequency(
            inst, picks, zscore_thresh, max_iter
        )

        # Mark channels as bad
        if bad_chs:
            for instance_name in apply_on:
                data[instance_name].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_name].info['bads']])

        data['preprocessing_steps'].append({
            'step': 'find_bads_channels_high_frequency',
            'instance': instance_name,
            'picks': picks_params,
            'apply_on': apply_on,
            'zscore_thresh': zscore_thresh,
            'max_iter': max_iter,
            'bad_channels': bad_chs,
            'n_bad_channels': len(bad_chs)
        })

        return data

    def _step_find_bads_epochs_threshold(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Find bad epochs using threshold-based rejection."""
        if 'epochs' not in data:
            raise ValueError("find_bads_epochs_threshold requires 'epochs' in data")

        picks_params = step_config.get('picks', None)
        reject = step_config.get('reject', {'eeg': 150e-6})
        n_channels_bad_epoch = step_config.get('n_channels_bad_epoch', 0.1)

        picks = self._get_picks(data['raw'].info, picks_params)

        bad_epochs = adaptive_reject.find_bads_epochs_threshold(
            data['epochs'], picks, reject, n_channels_bad_epoch
        )

        # Drop bad epochs
        if len(bad_epochs) > 0:
            data['epochs'].drop(bad_epochs, reason='ADAPTIVE AUTOREJECT')

        data['preprocessing_steps'].append({
            'step': 'find_bads_epochs_threshold',
            'picks': picks_params,
            'apply_on': ['epochs'], # only for compatibility with others reject steps
            'reject': reject,
            'n_channels_bad_epoch': n_channels_bad_epoch,
            'bad_epochs': bad_epochs.tolist() if hasattr(bad_epochs, 'tolist') else list(bad_epochs),
            'n_bad_epochs': len(bad_epochs),
            'n_epochs_remaining': len(data['epochs'])
        })

        return data

    def _step_save_clean_epochs(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Save clean epochs to a BIDS-derivatives-compatible path."""
        if 'epochs' not in data:
            raise ValueError("save_clean_epochs requires 'epochs' in data")

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

        # Ensure directory exists
        bids_path.mkdir(exist_ok=True)

        # Save epochs
        data['epochs'].save(bids_path.fpath, overwrite=overwrite)

        # Store paths & metadata in the data dict
        data['epochs_file'] = str(bids_path)
        data['n_epochs'] = len(data['epochs'])
        return data

    def _step_generate_json_report(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON reports."""

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
            report['raw'] = dict(
                n_channels=data['raw'].info.get('nchan'),
                sfreq=data['raw'].info.get('sfreq'),
                n_times=data['raw'].n_times
            )

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

        # Ensure directory exists
        bids_path.mkdir(exist_ok=True)

        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(bids_path.fpath, 'w') as f:
            json.dump(report, f, indent=2, cls=NpEncoder)

        data['json_report'] = str(bids_path)
        return data

    def _step_generate_html_report(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate HTML reports."""

        html_report = mne.Report(title=f'Preprocessing Report - Subject {data["subject"]}')

        if 'ica' in data:
            try:
                html_report.add_ica(ica=data['ica'], title='ICA Components', inst=data.get('raw'))
            except Exception:
                # If adding ICA fails, continue without stopping the pipeline
                pass

        if 'epochs' in data and data['epochs'] is not None:
            try:
                html_report.add_epochs(epochs=data['epochs'], title='Clean Epochs')
                evoked = data['epochs'].average()
                html_report.add_evokeds(evokeds=evoked, titles='Average Evoked Response')
            except Exception:
                pass

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

        # Ensure directory exists
        bids_path.mkdir(exist_ok=True)

        html_report.save(bids_path.fpath, overwrite=True, open_browser=False)

        data['html_report'] = str(bids_path)
        return data

    def _process_single_recording(self, bids_path: BIDSPath, progress: Progress = None, task_id: int = None) -> Dict[str, Any]:
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
        logger.info(f"Reading BIDS data from {bids_path.fpath}")
        data['raw'] = read_raw_bids(bids_path=bids_path)

        # Get pipeline steps from config
        pipeline_steps = self._get_pipeline_steps()

        # Execute each step in order
        for step_idx, step in enumerate(pipeline_steps):
            step_name = step.get('name')
            if step_name not in self.step_functions:
                raise ValueError(f"Unknown step '{step_name}' in pipeline execution")

            # Update progress for this step
            if progress and task_id is not None:
                progress.update(task_id, description=f"[cyan]Step: {step_name}", completed=step_idx)
            
            logger.info(f"Executing step: {step_name}")
            
            # Execute the step with its configuration
            step_config = {k: v for k, v in step.items() if k != 'name'}
            data = self.step_functions[step_name](data, step_config)

        # Mark as complete
        if progress and task_id is not None:
            progress.update(task_id, completed=len(pipeline_steps))

        # Prepare results
        results = {
            'subject': data.get('subject'),
            'task': data.get('task'),
            'session': data.get('session'),
            'acquisition': data.get('acquisition'),
            'run': data.get('run'),
            'raw_file': str(bids_path.fpath)
        }

        # Copy relevant output information to results
        for key in ['raw_file', 'epochs_file', 'json_report', 'html_report', 'n_epochs', 'preprocessing_steps']:
            if key in data:
                results[key] = data[key]

        logger.info(f"Successfully processed {bids_path.basename}")
        return results

    def run_pipeline(
        self, 
        subjects: Union[str, List[str]] = None,
        sessions: Union[str, List[str]] = None,
        tasks: Union[str, List[str]] = None,
        acquisitions: Union[str, List[str]] = None,
        runs: Union[str, List[str]] = None,
        extensions: Union[str, List[str]] = None,
        processings: Union[str, List[str]] = None,
        recordings: Union[str, List[str]] = None,
        spaces: Union[str, List[str]] = None,
        splits: Union[str, List[str]] = None,
        descriptions: Union[str, List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline using mne-bids find_matching_paths to query files.

        Parameters
        ----------
        subjects : str | list of str | None
            Subject ID(s) to process. None matches all subjects.
        sessions : str | list of str | None
            Session ID(s) to process. None matches all sessions.
        tasks : str | list of str | None
            Task(s) to process. None matches all tasks.
        acquisitions : str | list of str | None
            Acquisition parameter(s). None matches all acquisitions.
        runs : str | list of str | None
            Run number(s). None matches all runs.
        extensions : str | list of str | None
            File extension(s). None matches all extensions.
        processings : str | list of str | None
            Processing label(s). None matches all processings.
        recordings : str | list of str | None
            Recording name(s). None matches all recordings.
        spaces : str | list of str | None
            Coordinate space(s). None matches all spaces.
        splits : str | list of str | None
            Split(s) of continuous recording. None matches all splits.
        descriptions : str | list of str | None
            Description(s). None matches all descriptions.

        Returns
        -------
        all_results : dict
            Dictionary mapping subject -> list of results for each matching file.
        """
        # Use find_matching_paths to get all matching files
        logger.info("Finding matching paths...")
        matching_paths = find_matching_paths(
            root=self.bids_root,
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            acquisitions=acquisitions,
            runs=runs,
            extensions=extensions,
            processings=processings,
            recordings=recordings,
            spaces=spaces,
            splits=splits,
            descriptions=descriptions,
            suffixes='eeg',
            datatypes='eeg',
        )

        # Convert to list to get count
        matching_paths = list(matching_paths)
        logger.info(f"Found {len(matching_paths)} matching file(s) to process")

        all_results = {}
        
        # Create progress bars for matched paths and preprocessing steps
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            
            # Overall progress for all recordings
            overall_task = progress.add_task(
                "[green]Processing recordings", 
                total=len(matching_paths)
            )
            
            # Process each matching path
            for path_idx, raw_path in enumerate(matching_paths):
                subject = raw_path.subject
                
                # Get pipeline steps for this recording's progress bar
                pipeline_steps = self._get_pipeline_steps()
                
                # Create a task for the current recording's steps
                recording_name = raw_path.basename
                step_task = progress.add_task(
                    f"[cyan]{recording_name}", 
                    total=len(pipeline_steps)
                )
                
                try:
                    results = self._process_single_recording(raw_path, progress, step_task)
                    all_results.setdefault(subject, []).append(results)
                    logger.info(f"Successfully completed {recording_name}")
                except Exception as exc:
                    # Do not stop the whole batch if one subject fails; capture the error
                    logger.error(f"Error processing {recording_name}: {str(exc)}")
                    all_results.setdefault(subject, []).append({'error': str(exc)})
                finally:
                    # Remove the step task after this recording is done
                    progress.remove_task(step_task)
                
                # Update overall progress
                progress.update(overall_task, completed=path_idx + 1)

        logger.info(f"Pipeline completed. Processed {len(matching_paths)} recording(s)")
        return all_results