#!/usr/bin/env python3
"""
General EEG Preprocessing Pipeline using MNE-BIDS

This script provides a comprehensive preprocessing pipeline for EEG data:
1. Reads data using MNE-BIDS
2. Applies preprocessing steps (filtering, artifact removal, epoching)
3. Saves outputs to three derivative folders:
   - clean_epochs/: Preprocessed epochs in .fif format
   - html_reports/: MNE HTML reports
   - json_reports/: JSON reports for downstream analysis
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import mne
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
from mne.preprocessing import ICA
import numpy as np


class EEGPreprocessingPipeline:
    """
    A general EEG preprocessing pipeline using MNE-BIDS.
    """
    
    def __init__(self, bids_root, derivatives_root=None, config=None):
        """
        Initialize the preprocessing pipeline.
        
        Parameters
        ----------
        bids_root : str or Path
            Path to the BIDS root directory.
        derivatives_root : str or Path, optional
            Path to the derivatives folder. If None, creates 'derivatives' 
            in the BIDS root.
        config : dict, optional
            Configuration dictionary with preprocessing parameters.
        """
        self.bids_root = Path(bids_root)
        self.derivatives_root = Path(derivatives_root) if derivatives_root else self.bids_root / "derivatives"
        
        # Create derivative folders
        self.clean_epochs_dir = self.derivatives_root / "clean_epochs"
        self.html_reports_dir = self.derivatives_root / "html_reports"
        self.json_reports_dir = self.derivatives_root / "json_reports"
        
        for folder in [self.clean_epochs_dir, self.html_reports_dir, self.json_reports_dir]:
            folder.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            'l_freq': 0.5,          # High-pass filter frequency
            'h_freq': 40.0,         # Low-pass filter frequency
            'epochs_tmin': -0.2,    # Epoch start time
            'epochs_tmax': 0.8,     # Epoch end time
            'baseline': (None, 0),  # Baseline correction window
            'reject_criteria': {    # Artifact rejection criteria (in volts for EEG)
                'eeg': 150e-6,      # 150 µV
            },
            'ica_n_components': 20, # Number of ICA components
            'ica_method': 'fastica', # ICA method
            'event_id': None,       # Event IDs (None = use all)
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
        
        self.preprocessing_info = {}
    
    def read_data(self, subject, session=None, task=None, run=None):
        """
        Read raw data using MNE-BIDS.
        
        Parameters
        ----------
        subject : str
            Subject ID.
        session : str, optional
            Session ID.
        task : str, optional
            Task name.
        run : str, optional
            Run number.
            
        Returns
        -------
        raw : mne.io.Raw
            Raw data object.
        bids_path : BIDSPath
            BIDS path object.
        """
        bids_path = BIDSPath(
            subject=subject,
            session=session,
            task=task,
            run=run,
            root=self.bids_root,
            datatype='eeg'
        )
        
        print(f"Reading data from: {bids_path}")
        raw = read_raw_bids(bids_path=bids_path, verbose=True)
        
        self.preprocessing_info['bids_path'] = str(bids_path)
        self.preprocessing_info['n_channels'] = len(raw.ch_names)
        self.preprocessing_info['sampling_rate'] = raw.info['sfreq']
        self.preprocessing_info['duration'] = raw.times[-1]
        
        return raw, bids_path
    
    def preprocess_raw(self, raw):
        """
        Apply preprocessing to raw data.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw data object.
            
        Returns
        -------
        raw : mne.io.Raw
            Preprocessed raw data.
        """
        # Load data
        raw.load_data()
        
        # Store original info
        self.preprocessing_info['original_n_samples'] = len(raw.times)
        
        # Apply bandpass filter
        print(f"Applying bandpass filter: {self.config['l_freq']}-{self.config['h_freq']} Hz")
        raw.filter(l_freq=self.config['l_freq'], h_freq=self.config['h_freq'], 
                   fir_design='firwin', verbose=True)
        
        self.preprocessing_info['filter_applied'] = {
            'l_freq': self.config['l_freq'],
            'h_freq': self.config['h_freq']
        }
        
        # Set EEG reference (average reference)
        print("Setting average reference")
        raw.set_eeg_reference('average', projection=True)
        raw.apply_proj()
        
        self.preprocessing_info['reference'] = 'average'
        
        return raw
    
    def apply_ica(self, raw):
        """
        Apply ICA for artifact removal.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw data object.
            
        Returns
        -------
        raw : mne.io.Raw
            Raw data with ICA applied.
        ica : mne.preprocessing.ICA
            Fitted ICA object.
        """
        print(f"Applying ICA with {self.config['ica_n_components']} components")
        
        # Fit ICA
        ica = ICA(
            n_components=self.config['ica_n_components'],
            method=self.config['ica_method'],
            random_state=42
        )
        
        ica.fit(raw)
        
        # Find and exclude bad components (using automatic detection)
        # EOG artifact detection
        eog_indices, eog_scores = [], []
        if 'eog' in raw or any('EOG' in ch for ch in raw.ch_names):
            try:
                eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=3.0)
                ica.exclude.extend(eog_indices)
            except Exception as e:
                print(f"Could not detect EOG artifacts: {e}")
        
        # ECG artifact detection
        ecg_indices, ecg_scores = [], []
        if 'ecg' in raw or any('ECG' in ch for ch in raw.ch_names):
            try:
                ecg_indices, ecg_scores = ica.find_bads_ecg(raw, threshold=3.0)
                ica.exclude.extend(ecg_indices)
            except Exception as e:
                print(f"Could not detect ECG artifacts: {e}")
        
        print(f"Excluding ICA components: {ica.exclude}")
        
        # Apply ICA
        raw = ica.apply(raw.copy())
        
        self.preprocessing_info['ica'] = {
            'n_components': self.config['ica_n_components'],
            'method': self.config['ica_method'],
            'excluded_components': ica.exclude,
            'eog_components': eog_indices,
            'ecg_components': ecg_indices
        }
        
        return raw, ica
    
    def create_epochs(self, raw, events=None, event_id=None):
        """
        Create epochs from raw data.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw data object.
        events : array, optional
            Events array. If None, will try to find events in the data.
        event_id : dict, optional
            Event ID dictionary. If None, uses config or all events.
            
        Returns
        -------
        epochs : mne.Epochs
            Epochs object.
        """
        # Find events if not provided
        if events is None:
            try:
                events = mne.find_events(raw, stim_channel='STI 014', verbose=True)
            except:
                try:
                    # Try to find any stim channel
                    stim_channels = mne.pick_types(raw.info, meg=False, stim=True)
                    if len(stim_channels) > 0:
                        events = mne.find_events(raw, verbose=True)
                    else:
                        print("No events found in the data. Creating fake events for demonstration.")
                        # Create fake events for demonstration (every 1 second)
                        events = mne.make_fixed_length_events(raw, duration=1.0)
                except Exception as e:
                    print(f"Could not find events: {e}")
                    print("Creating fake events for demonstration.")
                    events = mne.make_fixed_length_events(raw, duration=1.0)
        
        # Use event_id from parameter, config, or create from events
        if event_id is None:
            event_id = self.config.get('event_id', None)
        
        if event_id is None:
            # Create event_id from unique event values
            unique_events = np.unique(events[:, 2])
            event_id = {f'event_{i}': i for i in unique_events}
        
        print(f"Creating epochs: tmin={self.config['epochs_tmin']}, tmax={self.config['epochs_tmax']}")
        print(f"Event ID: {event_id}")
        
        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=self.config['epochs_tmin'],
            tmax=self.config['epochs_tmax'],
            baseline=self.config['baseline'],
            reject=self.config['reject_criteria'],
            preload=True,
            verbose=True
        )
        
        self.preprocessing_info['epochs'] = {
            'n_epochs_before_rejection': len(epochs),
            'n_events': len(events),
            'event_id': event_id,
            'tmin': self.config['epochs_tmin'],
            'tmax': self.config['epochs_tmax'],
            'baseline': self.config['baseline']
        }
        
        # Drop bad epochs
        epochs.drop_bad()
        
        self.preprocessing_info['epochs']['n_epochs_after_rejection'] = len(epochs)
        self.preprocessing_info['epochs']['rejection_rate'] = (
            1 - len(epochs) / self.preprocessing_info['epochs']['n_epochs_before_rejection']
        ) * 100
        
        return epochs
    
    def save_epochs(self, epochs, subject, session=None, task=None, run=None):
        """
        Save epochs to .fif format.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Epochs to save.
        subject : str
            Subject ID.
        session : str, optional
            Session ID.
        task : str, optional
            Task name.
        run : str, optional
            Run number.
            
        Returns
        -------
        output_path : Path
            Path to saved epochs file.
        """
        # Create filename
        filename_parts = [f"sub-{subject}"]
        if session:
            filename_parts.append(f"ses-{session}")
        if task:
            filename_parts.append(f"task-{task}")
        if run:
            filename_parts.append(f"run-{run}")
        filename_parts.append("epo.fif")
        
        filename = "_".join(filename_parts)
        output_path = self.clean_epochs_dir / filename
        
        print(f"Saving epochs to: {output_path}")
        epochs.save(output_path, overwrite=True)
        
        self.preprocessing_info['output_epochs'] = str(output_path)
        
        return output_path
    
    def generate_html_report(self, raw, epochs, ica, subject, session=None, task=None, run=None):
        """
        Generate HTML report using MNE Report.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw data object.
        epochs : mne.Epochs
            Epochs object.
        ica : mne.preprocessing.ICA
            ICA object.
        subject : str
            Subject ID.
        session : str, optional
            Session ID.
        task : str, optional
            Task name.
        run : str, optional
            Run number.
            
        Returns
        -------
        report_path : Path
            Path to saved HTML report.
        """
        # Create filename
        filename_parts = [f"sub-{subject}"]
        if session:
            filename_parts.append(f"ses-{session}")
        if task:
            filename_parts.append(f"task-{task}")
        if run:
            filename_parts.append(f"run-{run}")
        filename_parts.append("report.html")
        
        filename = "_".join(filename_parts)
        report_path = self.html_reports_dir / filename
        
        print(f"Generating HTML report: {report_path}")
        
        # Create report
        report = mne.Report(title=f"Preprocessing Report - Subject {subject}", verbose=True)
        
        # Add raw data info
        report.add_raw(raw=raw, title='Raw Data', psd=True)
        
        # Add ICA info
        if ica:
            report.add_ica(
                ica=ica,
                title='ICA Components',
                inst=raw
            )
        
        # Add epochs info
        report.add_epochs(epochs=epochs, title='Clean Epochs')
        
        # Add evoked responses
        evoked = epochs.average()
        report.add_evokeds(evokeds=evoked, titles='Average Evoked Response')
        
        # Save report
        report.save(report_path, overwrite=True, open_browser=False)
        
        self.preprocessing_info['output_html_report'] = str(report_path)
        
        return report_path
    
    def generate_json_report(self, subject, session=None, task=None, run=None):
        """
        Generate JSON report for downstream analysis.
        
        Parameters
        ----------
        subject : str
            Subject ID.
        session : str, optional
            Session ID.
        task : str, optional
            Task name.
        run : str, optional
            Run number.
            
        Returns
        -------
        report_path : Path
            Path to saved JSON report.
        """
        # Create filename
        filename_parts = [f"sub-{subject}"]
        if session:
            filename_parts.append(f"ses-{session}")
        if task:
            filename_parts.append(f"task-{task}")
        if run:
            filename_parts.append(f"run-{run}")
        filename_parts.append("report.json")
        
        filename = "_".join(filename_parts)
        report_path = self.json_reports_dir / filename
        
        print(f"Generating JSON report: {report_path}")
        
        # Add metadata
        report_data = {
            'preprocessing_date': datetime.now().isoformat(),
            'pipeline_version': '1.0.0',
            'subject': subject,
            'session': session,
            'task': task,
            'run': run,
            'preprocessing_info': self.preprocessing_info,
            'config': self.config
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.preprocessing_info['output_json_report'] = str(report_path)
        
        return report_path
    
    def run_pipeline(self, subject, session=None, task=None, run=None, 
                    apply_ica=True, save_outputs=True):
        """
        Run the complete preprocessing pipeline.
        
        Parameters
        ----------
        subject : str
            Subject ID.
        session : str, optional
            Session ID.
        task : str, optional
            Task name.
        run : str, optional
            Run number.
        apply_ica : bool, optional
            Whether to apply ICA. Default is True.
        save_outputs : bool, optional
            Whether to save outputs. Default is True.
            
        Returns
        -------
        results : dict
            Dictionary with preprocessing results.
        """
        print("="*80)
        print(f"Starting EEG Preprocessing Pipeline")
        print(f"Subject: {subject}, Session: {session}, Task: {task}, Run: {run}")
        print("="*80)
        
        # Reset preprocessing info
        self.preprocessing_info = {}
        
        # 1. Read data
        raw, bids_path = self.read_data(subject, session, task, run)
        
        # 2. Preprocess raw data
        raw = self.preprocess_raw(raw)
        
        # 3. Apply ICA (optional)
        ica = None
        if apply_ica:
            raw, ica = self.apply_ica(raw)
        
        # 4. Create epochs
        epochs = self.create_epochs(raw)
        
        # 5. Save outputs
        if save_outputs:
            epochs_path = self.save_epochs(epochs, subject, session, task, run)
            html_report_path = self.generate_html_report(raw, epochs, ica, subject, session, task, run)
            json_report_path = self.generate_json_report(subject, session, task, run)
        
        print("="*80)
        print("Preprocessing completed successfully!")
        print(f"Clean epochs saved to: {self.clean_epochs_dir}")
        print(f"HTML reports saved to: {self.html_reports_dir}")
        print(f"JSON reports saved to: {self.json_reports_dir}")
        print("="*80)
        
        results = {
            'raw': raw,
            'epochs': epochs,
            'ica': ica,
            'preprocessing_info': self.preprocessing_info
        }
        
        return results


def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(
        description='EEG Preprocessing Pipeline using MNE-BIDS'
    )
    parser.add_argument(
        '--bids-root',
        type=str,
        required=True,
        help='Path to BIDS root directory'
    )
    parser.add_argument(
        '--subject',
        type=str,
        required=True,
        help='Subject ID'
    )
    parser.add_argument(
        '--session',
        type=str,
        default=None,
        help='Session ID'
    )
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        help='Task name'
    )
    parser.add_argument(
        '--run',
        type=str,
        default=None,
        help='Run number'
    )
    parser.add_argument(
        '--derivatives-root',
        type=str,
        default=None,
        help='Path to derivatives folder (default: BIDS_ROOT/derivatives)'
    )
    parser.add_argument(
        '--no-ica',
        action='store_true',
        help='Skip ICA application'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize pipeline
    pipeline = EEGPreprocessingPipeline(
        bids_root=args.bids_root,
        derivatives_root=args.derivatives_root,
        config=config
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(
        subject=args.subject,
        session=args.session,
        task=args.task,
        run=args.run,
        apply_ica=not args.no_ica
    )
    
    print("\nPipeline execution completed successfully!")


if __name__ == '__main__':
    main()
