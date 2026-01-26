#!/usr/bin/env python3
"""
EEG Preprocessing Report Generator.

This module provides functionality to generate JSON and HTML reports from
preprocessed EEG data saved by the preprocessing pipeline. It can load
intermediate results from disk and generate comprehensive reports without
requiring the full preprocessing to be re-run.

Main Components
---------------
- ReportGenerator: Class for generating reports from saved preprocessing results
  - Loads intermediate results from disk (metadata + data objects)
  - Generates JSON reports with preprocessing summary
  - Generates interactive HTML reports with visualizations
  - Can process multiple subjects/sessions independently

Usage Example
-------------
```python
from report_generator import ReportGenerator

# Initialize report generator
generator = ReportGenerator(
    bids_root='/path/to/bids',
    intermediate_results_path='/path/to/intermediate/results'
)

# Generate reports for specific recording
generator.generate_reports()
```

See README.md for detailed documentation and examples.
"""
from pathlib import Path
from typing import Union, Dict, Any
import json
import pickle
import mne
from mne.utils import logger
import numpy as np
from mne_bids import BIDSPath
from utils import NpEncoder
import matplotlib.pyplot as plt


class ReportGenerator:
    """
    Generate JSON and HTML reports from saved preprocessing results.
    
    This class loads intermediate results saved by the preprocessing pipeline
    and generates comprehensive reports including visualizations and summaries.
    
    Parameters
    ----------
    bids_root : str | Path
        Path to BIDS root directory
    intermediate_results_path : str | Path | None
        Path to intermediate results directory. If None, uses default location:
        bids_root/derivatives/nice_preprocessing/intermediate/<recording_id>/
    """
    
    def __init__(self, bids_root: Union[str, Path], intermediate_results_path: Union[str, Path] = None):
        self.bids_root = Path(bids_root)
        self.intermediate_results_path = Path(intermediate_results_path) if intermediate_results_path else None
        
    def load_intermediate_results(self, intermediate_results_path: Union[str, Path] = None) -> Dict[str, Any]:
        """
        Load intermediate preprocessing results from disk.
        
        Parameters
        ----------
        intermediate_results_path : str | Path | None
            Path to intermediate results directory. If None, uses self.intermediate_results_path
            
        Returns
        -------
        data : dict
            Dictionary containing:
            - metadata (subject, task, session, acquisition, preprocessing_steps, events, event_id)
            - raw (MNE Raw object, if saved)
            - epochs (MNE Epochs object, if saved)
            - ica (MNE ICA object, if saved)
        """
        if intermediate_results_path:
            results_path = Path(intermediate_results_path)
        elif self.intermediate_results_path:
            results_path = self.intermediate_results_path
        else:
            raise ValueError("intermediate_results_path must be provided either in __init__ or load_intermediate_results")
        
        if not results_path.exists():
            raise FileNotFoundError(f"Intermediate results path not found: {results_path}")
        
        logger.info(f"Loading intermediate results from: {results_path}")
        
        # Load metadata
        metadata_file = results_path / 'metadata.json'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"  Loaded metadata for subject {metadata['subject']}, task {metadata['task']}")
        
        # Initialize data dictionary with metadata
        data = {
            'subject': metadata['subject'],
            'task': metadata['task'],
            'session': metadata.get('session', None),
            'acquisition': metadata.get('acquisition', None),
            'preprocessing_steps': metadata.get('preprocessing_steps', []),
        }
        
        # Add events if available
        if 'events' in metadata:
            data['events'] = np.array(metadata['events'])
            data['events_sfreq'] = metadata.get('events_sfreq', None)
        if 'event_id' in metadata:
            data['event_id'] = metadata['event_id']
        
        # Load pickled objects
        raw_file = results_path / 'raw.pkl'
        if raw_file.exists():
            with open(raw_file, 'rb') as f:
                data['raw'] = pickle.load(f)
            logger.info(f"  Loaded raw object")
        
        epochs_file = results_path / 'epochs.pkl'
        if epochs_file.exists():
            with open(epochs_file, 'rb') as f:
                data['epochs'] = pickle.load(f)
            logger.info(f"  Loaded epochs object")
        
        ica_file = results_path / 'ica.pkl'
        if ica_file.exists():
            with open(ica_file, 'rb') as f:
                data['ica'] = pickle.load(f)
            logger.info(f"  Loaded ICA object")
        
        logger.info(f"Successfully loaded intermediate results")
        return data
    
    def generate_json_report(self, data: Dict[str, Any]) -> str:
        """
        Generate JSON report from preprocessing data.
        
        Parameters
        ----------
        data : dict
            Data dictionary containing preprocessing results
            
        Returns
        -------
        report_path : str
            Path to saved JSON report
        """
        # JSON report
        report = {
            'subject': data['subject'],
            'task': data['task'],
            'session': data.get('session', None),
            'acquisition': data.get('acquisition', None),
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

        with open(bids_path.fpath, 'w') as f:
            json.dump(report, f, indent=2, cls=NpEncoder)

        logger.info(f"JSON report saved to: {bids_path.fpath}")
        return str(bids_path.fpath)
    
    def generate_html_report(
        self, 
        data: Dict[str, Any],
        picks_params: Any = None,
        excluded_channels: Any = None,
        compare_instances: list = None,
        plot_raw_kwargs: dict = None,
        plot_ica_kwargs: dict = None,
        plot_events_kwargs: dict = None,
        plot_epochs_kwargs: dict = None,
        plot_evokeds_kwargs: dict = None,
        n_time_points: int = None
    ) -> str:
        """
        Generate HTML report from preprocessing data.
        
        Parameters
        ----------
        data : dict
            Data dictionary containing preprocessing results
        picks_params : list | None
            Channel type picks (e.g., ['eeg'])
        excluded_channels : list | None
            Channels to exclude from visualization
        compare_instances : list
            List of instance comparisons to visualize
        plot_raw_kwargs : dict
            Keyword arguments for raw plotting
        plot_ica_kwargs : dict
            Keyword arguments for ICA plotting
        plot_events_kwargs : dict
            Keyword arguments for events plotting
        plot_epochs_kwargs : dict
            Keyword arguments for epochs plotting
        plot_evokeds_kwargs : dict
            Keyword arguments for evokeds plotting
        n_time_points : int | None
            Number of time points for evoked plots
            
        Returns
        -------
        report_path : str
            Path to saved HTML report
        """
        from report import (
            collect_bad_channels_from_steps,
            create_bad_channels_topoplot,
            create_preprocessing_steps_table
        )

        # Set defaults
        compare_instances = compare_instances or []
        plot_raw_kwargs = plot_raw_kwargs or {}
        plot_ica_kwargs = plot_ica_kwargs or {}
        plot_events_kwargs = plot_events_kwargs or {}
        plot_epochs_kwargs = plot_epochs_kwargs or {}
        plot_evokeds_kwargs = plot_evokeds_kwargs or {}

        if 'preprocessing_steps' not in data:
            raise ValueError("generate_html_report requires 'preprocessing_steps' in data")
        elif not isinstance(data['preprocessing_steps'], list):
            raise ValueError("data['preprocessing_steps'] must be a list")

        # Get info from epochs if available, otherwise from raw
        inst = data['raw'] if 'raw' in data else data['epochs'] if 'epochs' in data else None
        if inst is None:
            raise ValueError("generate_html_report requires either 'raw' or 'epochs' in data")

        # Compute picks for channel selection
        picks = self._get_picks(inst.info, picks_params, excluded_channels)

        preprocessing_steps = data['preprocessing_steps']

        html_report = mne.Report(title=f'Preprocessing Report - Subject {data["subject"]}')

        # Add bad channels topoplot section
        bad_channels = collect_bad_channels_from_steps(preprocessing_steps)

        # Create topoplot if we have bad channels and info
        if len(bad_channels) > 0:
            logger.info(f"Adding bad channels topoplot with {len(bad_channels)} bad channels")
            fig = create_bad_channels_topoplot(inst.info, bad_channels)

            if fig is not None:
                # Add to report
                html_report.add_figure(
                    fig=fig,
                    title='Bad Channels',
                    caption=f'Topoplot showing {len(bad_channels)} bad channels marked with red crosses'
                )
                plt.close(fig)

        # Add preprocessing steps table section
        html_content = create_preprocessing_steps_table(data['preprocessing_steps'])

        # ---------- Preprocessing steps ----------
        if html_content is not None:
            # Add the HTML table to the report
            html_report.add_html(
                html=html_content,
                title='Preprocessing Steps',
            )

        # ---------- ICA ----------
        if data.get('ica', None) is not None:

            section = "ICA"

            html_report.add_ica(
                ica=data['ica'],
                title='ICA Components',
                inst=None,
                **plot_ica_kwargs
            )

            ica_step = [step for step in preprocessing_steps if step['step'] == 'ica']
            ica_step = ica_step[-1] if len(ica_step) > 0 else {}
            eog_step_report = ica_step.get('eog_detection', {})
            eog_idx = eog_step_report.get('eog_excluded_components', []) or []
            eog_scores = eog_step_report.get('eog_scores', None)
            ecg_step_report = ica_step.get('ecg_detection', {})
            ecg_idx = ecg_step_report.get('ecg_excluded_components', [])
            ecg_scores = ecg_step_report.get('ecg_scores', None)

            if len(eog_idx) > 0:

                if eog_scores is not None:
                    scores = np.array(eog_scores, dtype=float)

                    if scores.ndim == 1:
                        scores = scores.reshape(-1, 1)  # Make it 2D for uniform processing

                    # Heatmap (EOG channels x ICA components)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    im = ax.imshow(scores, aspect="auto", origin="lower")

                    n_components = scores.shape[1]

                    # X axis: ICA components as discrete labels 1..N
                    ax.set_xticks(np.arange(n_components))
                    ax.set_xticklabels(np.arange(n_components))

                    ax.set_xlabel("ICA component")
                    ax.set_ylabel("EOG channel")

                    eog_names = (
                        eog_step_report.get("eog_channels_present", None)
                        or eog_step_report.get("eog_channels_requested", None)
                        or []
                    )
                    if isinstance(eog_names, list) and len(eog_names) == scores.shape[0]:
                        ax.set_yticks(np.arange(len(eog_names)))
                        ax.set_yticklabels(eog_names)

                    ax.set_title("EOG scores (per EOG channel × ICA component)")

                    fig.colorbar(im, ax=ax, shrink=0.8, label="EOG score")

                    html_report.add_figure(
                        fig=fig,
                        title="ICA - EOG scores heatmap",
                        section='ICA - EOG'
                    )
                    plt.close(fig)

                    # Aggregate to 1 score per component for barplot
                    scores_1d = np.max(np.abs(scores), axis=0)

                    # Barplot (always 1D after aggregation if needed)
                    fig1 = plt.figure()
                    ax = fig1.add_subplot(111)
                    ax.bar(np.arange(len(scores_1d)), scores_1d)
                    ax.set_xlabel("ICA component")
                    ax.set_ylabel("max |EOG score| across EOG channels" if (eog_scores is not None and np.array(eog_scores).ndim == 2) else "|EOG score|")
                    ax.set_title(f"EOG scores (selected: {eog_idx})")
                    html_report.add_figure(
                        fig=fig1,
                        title="ICA - EOG scores",
                        section='ICA - EOG'
                    )
                    plt.close(fig1)
            
            if len(ecg_idx) > 0:

                if ecg_scores is not None:
                    scores = np.array(ecg_scores, dtype=float)

                    if scores.ndim == 1:
                        scores = scores.reshape(-1, 1)  # Make it 2D for uniform processing

                    # Heatmap (ECG channels x ICA components)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    im = ax.imshow(scores, aspect="auto", origin="lower")

                    n_components = scores.shape[1]

                    # X axis: ICA components as discrete labels 1..N
                    ax.set_xticks(np.arange(n_components))
                    ax.set_xticklabels(np.arange(n_components))

                    ax.set_xlabel("ICA component")
                    ax.set_ylabel("ECG channel")

                    ecg_names = (
                        ecg_step_report.get("ecg_channels_present", None)
                        or ecg_step_report.get("ecg_channels_requested", None)
                        or []
                    )
                    if isinstance(ecg_names, list) and len(ecg_names) == scores.shape[0]:
                        ax.set_yticks(np.arange(len(ecg_names)))
                        ax.set_yticklabels(ecg_names)

                    ax.set_title("ECG scores (per ECG channel × ICA component)")
                    fig.colorbar(im, ax=ax, shrink=0.8, label="ECG score")

                    html_report.add_figure(
                        fig=fig,
                        title="ICA - ECG scores heatmap",
                        section='ICA - ECG'
                    )
                    plt.close(fig)

                    # Aggregate to 1 score per component for barplot
                    scores_1d = np.max(np.abs(scores), axis=0)

                    # Barplot (always 1D after aggregation if needed)
                    fig1 = plt.figure()
                    ax = fig1.add_subplot(111)
                    ax.bar(np.arange(len(scores_1d)), scores_1d)
                    ax.set_xlabel("ICA component")
                    ax.set_ylabel("max |ECG score| across ECG channels" if (ecg_scores is not None and np.array(ecg_scores).ndim == 2) else "|ECG score|")
                    ax.set_title(f"ECG scores (selected: {ecg_idx})")
                    html_report.add_figure(
                        fig=fig1,
                        title="ICA - ECG scores",
                        section='ICA - ECG'
                    )
                    plt.close(fig1)

        # ---------- Compare instances preprocessing (full recording) ----------
        for contrast in compare_instances:
            inst_a_name = contrast['instance_a']['name']
            inst_a_label = contrast['instance_a']['label']
            inst_b_name = contrast['instance_b']['name']
            inst_b_label = contrast['instance_b']['label']

            if inst_a_name not in data or inst_b_name not in data:
                raise ValueError(f"compare_instances step requires both '{inst_a_name}' and '{inst_b_name}' in data")

            inst_a = data[inst_a_name]
            inst_b = data[inst_b_name]

            # Ensure channel alignment (same channel order)
            ch_names_picks = self._get_picks(
                inst.info,
                picks_params,
                excluded_channels
            )
            ch_names_a = sorted([inst_a.ch_names[pick] for pick in ch_names_picks])
            ch_names_b = sorted([inst_b.ch_names[pick] for pick in ch_names_picks])
            if set(ch_names_a) != set(ch_names_b):
                raise ValueError(f"compare_instances step: channel mismatch between '{inst_a}' and '{inst_b}' after picking")

            raw_b = inst_b.copy().pick(picks=ch_names_picks).reorder_channels(ch_names_a)
            raw_a = inst_a.copy().pick(picks=ch_names_picks).reorder_channels(ch_names_a)

            Xb = raw_b.get_data()
            Xa = raw_a.get_data()
            times = raw_a.times

            # Metrics over full recording
            gfp_b = np.std(Xb, axis=0)
            gfp_a = np.std(Xa, axis=0)

            mean_b = np.mean(Xb, axis=0)
            mean_a = np.mean(Xa, axis=0)

            diff_abs = np.mean(np.abs(Xb - Xa), axis=0)

            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

            axes[0].plot(times, gfp_b, color='red', alpha=0.35, label=inst_b_label)
            axes[0].plot(times, gfp_a, color='black', linewidth=1.0, label=inst_a_label)
            axes[0].set_title('EEG Global Field Power (full recording)')
            axes[0].legend(loc='upper right')

            axes[1].plot(times, mean_b, color='red', alpha=0.35, label=inst_b_label)
            axes[1].plot(times, mean_a, color='black', linewidth=1.0, label=inst_a_label)
            axes[1].set_title('Mean EEG across channels (full recording)')
            axes[1].legend(loc='upper right')

            axes[2].plot(times, diff_abs, color='purple', linewidth=1.0)
            axes[2].set_title(f'Mean absolute difference |{inst_a_label} - {inst_b_label}| (full recording)')
            axes[2].set_xlabel('Time (s)')

            fig.tight_layout()
            html_report.add_figure(
                fig=fig,
                title=contrast['title'],
                section='Contrasts'
            )
            plt.close(fig)

        # ---------- Cleaned Raw report ----------
        if data.get('raw', None) is not None:
            html_report.add_raw(
                raw=data['raw'].copy().pick(picks=picks),
                title='Clean Raw Data',
                **plot_raw_kwargs
            )

        # ---------- Events report ----------
        if 'events' in data and data['events'] is not None:
            html_report.add_events(
                events=data['events'],
                event_id=data.get('event_id', None),
                sfreq=data['events_sfreq'],
                title='Found Events',
                **plot_events_kwargs
            )

        # ---------- Cleaned Epochs report ----------
        if data.get('epochs', None) is not None:

            epochs=data['epochs'].copy().pick(picks=picks)

            html_report.add_epochs(
                epochs=epochs,
                title='Clean Epochs',
                **plot_epochs_kwargs
            )
            
            html_report.add_evokeds(
                evokeds=epochs.average(by_event_type=True),
                n_time_points=n_time_points,
                **plot_evokeds_kwargs
            )

        # Derivatives root for this pipeline
        deriv_root = self.bids_root / "derivatives" / "nice_preprocessing" / "reports"

        bids_path = BIDSPath(
            subject=data['subject'],
            task=data['task'],
            session=data.get('session', None),
            acquisition=data.get('acquisition', None),
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

        logger.info(f"HTML report saved to: {bids_path.fpath}")
        return str(bids_path.fpath)
    
    def _get_picks(self, info, picks_params=None, excluded_channels=None):
        """Get channel picks for plotting."""
        if picks_params is None:
            picks = mne.pick_types(info, eeg=True, exclude=[])
        else:
            picks = mne.pick_types(info, **{k: True for k in picks_params}, exclude=[])
        
        if excluded_channels:
            excluded_idx = [info.ch_names.index(ch) for ch in excluded_channels if ch in info.ch_names]
            picks = [p for p in picks if p not in excluded_idx]
        
        return picks
    
    def generate_reports(
        self,
        intermediate_results_path: Union[str, Path] = None,
        picks_params: Any = None,
        excluded_channels: Any = None,
        compare_instances: list = None,
        plot_raw_kwargs: dict = None,
        plot_ica_kwargs: dict = None,
        plot_events_kwargs: dict = None,
        plot_epochs_kwargs: dict = None,
        plot_evokeds_kwargs: dict = None,
        n_time_points: int = None
    ) -> Dict[str, str]:
        """
        Load intermediate results and generate both JSON and HTML reports.
        
        Parameters
        ----------
        intermediate_results_path : str | Path | None
            Path to intermediate results directory
        picks_params : list | None
            Channel type picks (e.g., ['eeg'])
        excluded_channels : list | None
            Channels to exclude from visualization
        compare_instances : list
            List of instance comparisons to visualize
        plot_raw_kwargs : dict
            Keyword arguments for raw plotting
        plot_ica_kwargs : dict
            Keyword arguments for ICA plotting
        plot_events_kwargs : dict
            Keyword arguments for events plotting
        plot_epochs_kwargs : dict
            Keyword arguments for epochs plotting
        plot_evokeds_kwargs : dict
            Keyword arguments for evokeds plotting
        n_time_points : int | None
            Number of time points for evoked plots
            
        Returns
        -------
        report_paths : dict
            Dictionary with 'json_report' and 'html_report' paths
        """
        # Load intermediate results
        data = self.load_intermediate_results(intermediate_results_path)
        
        # Generate reports
        json_path = self.generate_json_report(data)
        html_path = self.generate_html_report(
            data,
            picks_params=picks_params,
            excluded_channels=excluded_channels,
            compare_instances=compare_instances,
            plot_raw_kwargs=plot_raw_kwargs,
            plot_ica_kwargs=plot_ica_kwargs,
            plot_events_kwargs=plot_events_kwargs,
            plot_epochs_kwargs=plot_epochs_kwargs,
            plot_evokeds_kwargs=plot_evokeds_kwargs,
            n_time_points=n_time_points
        )
        
        return {
            'json_report': json_path,
            'html_report': html_path
        }
