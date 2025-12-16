#!/usr/bin/env python3
"""
EEG preprocessing pipeline using MNE-BIDS.

This version is modular with separate functions for each preprocessing step.
The pipeline is configuration-driven - you specify steps, their order, and parameters.
"""
from itertools import product
from pathlib import Path
from typing import Union, Dict, Any, List
import json
import mne
from mne.utils import logger
import numpy as np
from mne_bids import BIDSPath, read_raw_bids
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import adaptive_reject
from collections import defaultdict



class EEGPreprocessingPipeline:
    def __init__(self, bids_root: Union[str, Path], output_root: Union[str, Path] = None, config: Dict[str, Any] = None):
        self.bids_root = Path(bids_root)
        self.config = config or {}

        # Map step names to their corresponding methods
        self.step_functions = {
            'strip_recording': self._step_strip_recording,
            'concatenate_recordings': self._step_concatenate_recordings,
            'set_montage': self._step_set_montage,
            'drop_unused_channels': self._step_drop_unused_channels,
            'bandpass_filter': self._step_bandpass_filter,
            'notch_filter': self._step_notch_filter,
            'resample': self._step_resample,
            'reference': self._step_reference,
            'interpolate_bad_channels': self._step_interpolate_bad_channels,
            'drop_bad_channels': self._step_drop_bad_channels,
            'ica': self._step_ica,
            'find_events': self._step_find_events,
            'epoch': self._step_epoch,
            'chunk_in_epoch': self._step_chunk_in_epoch,
            'find_flat_channels': self._step_find_flat_channels,
            'find_bads_channels_threshold': self._step_find_bads_channels_threshold,
            'find_bads_channels_variance': self._step_find_bads_channels_variance,
            'find_bads_channels_high_frequency': self._step_find_bads_channels_high_frequency,
            'find_bads_epochs_threshold': self._step_find_bads_epochs_threshold,
            'save_clean_instance': self._step_save_clean_instance,
            'generate_json_report': self._step_generate_json_report,
            'generate_html_report': self._step_generate_html_report,
        }

        # Validate pipeline steps if provided in config
        pipeline_cfg = self.config.get('pipeline', [])
        unknown = [s.get('name') for s in pipeline_cfg if s.get('name') not in self.step_functions]
        if unknown:
            raise ValueError(f"Unknown pipeline steps in config: {unknown}")

    def _get_entity_values(self, entity_key: str, entity_value: any) -> List[Union[str, None]]:
        """Get all unique values for a given BIDS entity in the dataset."""
        if isinstance(entity_value, str):
            return [entity_value]
    
        if isinstance(entity_value, list):
            return entity_value
        
        # TODO: restore get_entity_vals and think how to improve the acq and other parameters that may be usefull to group but not mandatory
        if entity_value is None:
            return [None]
    
        raise ValueError(f"Invalid type for entity '{entity_key}': {type(entity_value)}")

    def _find_events_from_raw(self, raw, get_events_from='annotations', shortest_event=1, event_id='auto', stim_channel=None):
        
        if get_events_from == 'stim_channel':
            events = mne.find_events(
                raw,
                shortest_event=shortest_event,
                stim_channel=stim_channel,
                verbose=False
            )
            return events, None
        
        if get_events_from == 'annotations':
            events, found_event_id = mne.events_from_annotations(raw, event_id=event_id)
            return events, found_event_id

        raise ValueError(f"Invalid get_events_from method: {get_events_from}")

    def _get_pipeline_steps(self) -> List[Dict[str, Any]]:
        """Retrieve the list of pipeline steps from the configuration."""
        pipeline_steps = self.config.get('pipeline', [])

        if not pipeline_steps:
            pipeline_steps = [
                {'name': 'load_data'},
                {'name': 'bandpass_filter', 'l_freq': 0.5, 'h_freq': 45.0},
                {'name': 'reference', 'type': 'average'},
                {'name': 'save_clean_instance', 'instance': 'epochs'},
                {'name': 'generate_json_report'},
                {'name': 'generate_html_report'},
            ]
    
        return pipeline_steps

    def _apply_excluded_channels(self, info: mne.Info, picks: List[int], excluded_channels: List[str] = None) -> List[int]:
        """
        Auxiliary function to exclude specific channels from picks.
        
        This function allows excluding channels (e.g., reference channels like 'Cz') 
        from analysis steps where it makes sense, to avoid reference problems.
        
        Parameters
        ----------
        info : mne.Info
            MNE info object containing channel information
        picks : list of int
            Channel indices to filter
        excluded_channels : list of str, optional
            List of channel names to exclude from picks
            
        Returns
        -------
        picks : list of int
            Filtered channel indices with excluded channels removed
        """
        if excluded_channels is None or len(excluded_channels) == 0:
            return picks
            
        # Get channel names for the picks
        ch_names = [info['ch_names'][pick] for pick in picks]
        
        # Filter out excluded channels
        filtered_picks = [pick for pick, ch_name in zip(picks, ch_names) 
                         if ch_name not in excluded_channels]
        
        logger.info(f"Excluding channels: {excluded_channels}. "
                   f"Reduced from {len(picks)} to {len(filtered_picks)} channels.")
        
        return filtered_picks

    def _get_picks(self, info: mne.Info, picks_params: Any, excluded_channels: List[str] = None) -> List[int]:
        """
        Get channel picks with optional exclusion of specific channels.
        
        Parameters
        ----------
        info : mne.Info
            MNE info object containing channel information
        picks_params : list, tuple, or None
            Channel type specification (e.g., ['eeg'], ['eeg', 'eog'])
        excluded_channels : list of str, optional
            List of channel names to exclude from picks
            
        Returns
        -------
        picks : list of int
            Channel indices, excluding 'bads' and any specified excluded_channels
        """
        # Compute picks if provided, otherwise return all EEG channels
        if isinstance(picks_params, (list, tuple)):
            picks = mne.pick_types(
                info,
                exclude='bads',
                **{ch_type: True for ch_type in picks_params}
            )
        else:
            picks = mne.pick_types(
                info,
                exclude='bads',
                eeg=True,
                eog=False,
                meg=False
            )
        
        # Apply excluded_channels filter
        picks = self._apply_excluded_channels(info, picks, excluded_channels)
        
        return picks

    def _step_strip_recording(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        
        instance = step_config.get('instance', 'raw')
        start_padding = step_config.get('start_padding', 1)
        end_padding = step_config.get('end_padding', 1)
        get_events_from = step_config.get('get_events_from', 'annotations')
        shortest_event = step_config.get('shortest_event', 1)
        event_id = step_config.get('event_id', 'auto')
        

        if instance not in data:
            raise ValueError(f"strip recordings step requires '{instance}' to be present in data (either 'all_raw', 'raw')")

        # TODO: improve this and make it general to all corresponding steps        
        all_instances = data[instance]
        if not isinstance(all_instances, list):
            all_instances = [all_instances]

        for i, inst in enumerate(all_instances):
            events, _ = self._find_events_from_raw(
                inst,
                get_events_from=get_events_from,
                shortest_event=shortest_event,
                event_id=event_id
            )
            
            start = inst.times[events[0,0]] - start_padding
            end = inst.times[events[-1,0]] + end_padding

            start = max(0, start)
            end = min(inst.times[-1], end)
            
            inst.crop(start, end)
            
            data['preprocessing_steps'].append({
                'step': 'strip_recording',
                'instance': f'{instance}-{i}',
                'start': start,
                'end': end
            })
        
        return data
    
    def _step_concatenate_recordings(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        if 'all_raw' not in data:
            raise ValueError("notch_filter requires 'all_raw' in data")

        if len(data['all_raw']) > 1:
            data['raw'] = mne.concatenate_raws(data['all_raw'])
        else:
            data['raw'] = data['all_raw'][0]
        
        data['preprocessing_steps'].append({
            'step': 'concatenate_recordings',
        })
        
        return data

    def _step_set_montage(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        if 'raw' not in data:
            raise ValueError("set_montage requires 'raw' in data")

        montage_name = step_config.get('montage', 'standard_1020')

        montage = mne.channels.make_standard_montage(montage_name)
        data['raw'].set_montage(montage, on_missing="ignore")

        data['preprocessing_steps'].append({
            'step': 'set_montage',
            'montage': montage_name
        })
        return data

    def _step_drop_unused_channels(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Drop unused channels from the data."""
        channels_to_drop = step_config.get('channels_to_drop', [])
        instance = step_config.get('instance', 'raw') 

        if instance not in data:
            raise ValueError(f"drop_unused_channels step requires '{instance}' to be present in data (either 'raw' or 'epochs')")

        data[instance].drop_channels(channels_to_drop)

        data['preprocessing_steps'].append({
            'step': 'drop_unused_channels',
            'instance': instance,
            'channels_dropped': channels_to_drop
        })

        return data

    def _step_bandpass_filter(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply bandpass filtering."""
        if 'raw' not in data:
            raise ValueError("bandpass_filter requires 'raw' in data")

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        l_freq = step_config.get('l_freq', 0.5)
        l_freq_order = step_config.get('l_freq_order', 6)
        h_freq = step_config.get('h_freq', 45.0)
        h_freq_order = step_config.get('h_freq_order', 8)
        n_jobs = step_config.get('n_jobs', 1)

        # Compute picks if provided, otherwise None (all channels)
        picks = self._get_picks(data['raw'].info, picks_params, excluded_channels)

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
                'excluded_channels': excluded_channels,
                'params': high_pass_filter_params
            },
            {
                'step': 'low_pass_filter',
                'picks': picks_params,
                'excluded_channels': excluded_channels,
                'params': low_pass_filter_params
            }
        ])

        return data

    def _step_notch_filter(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply notch filtering."""
        if 'raw' not in data:
            raise ValueError("notch_filter requires 'raw' in data")

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        freqs = step_config.get('freqs', [50.0, 100.0])
        notch_widths = step_config.get('notch_widths', None)
        method = step_config.get('method', 'fft')
        n_jobs = step_config.get('n_jobs', 1)

        # Compute picks if provided
        picks = self._get_picks(data['raw'].info, picks_params, excluded_channels)

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
            'excluded_channels': excluded_channels,
            'freqs': freqs,
            'method': method,
            'notch_widths': notch_widths
        })

        return data

    def _step_resample(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Resample the data."""
        instance = step_config.get('instance', 'raw')
        
        if instance not in data:
            raise ValueError(f"resample requires '{instance}' in data")

        resample_events = step_config.get('resample_events', False)
        sfreq = step_config.get('sfreq', 250)
        npad = step_config.get('npad', 'auto')
        n_jobs = step_config.get('n_jobs', 1)

        data[instance].resample(
            sfreq=sfreq,
            npad=npad,
            n_jobs=n_jobs
        )

        if resample_events and 'events' in data:
            mne.events.resample_events(
                data['events'],
                data['events_sfreq'],
                sfreq
            ) 

        # Store info for reporting
        data['preprocessing_steps'].append({
            'step': 'resample',
            'instance': instance,
            'resample_events': resample_events,
            'sfreq': sfreq,
            'npad': npad,
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

    def _step_interpolate_bad_channels(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Interpolate bad channels."""
        instance = step_config.get('instance', 'epochs')
        excluded_channels = step_config.get('excluded_channels', None)

        if instance not in data:
            raise ValueError(f"interpolate_bad_channels step requires '{instance}' to be present in data (either 'raw' or 'epochs')")

        data[instance].interpolate_bads(
            reset_bads=True,
            exclude=excluded_channels
        )

        data['preprocessing_steps'].append({
            'step': 'interpolate_bad_channels',
            'excluded_channels': excluded_channels,
            'instance': instance
        })

        return data

    def _step_drop_bad_channels(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Drop bad channels without interpolation."""
        instance = step_config.get('instance', 'epochs')
        excluded_channels = step_config.get('excluded_channels', None)

        if instance not in data:
            raise ValueError(f"drop_bad_channels step requires '{instance}' to be present in data (either 'raw' or 'epochs')")

        # Get the list of bad channels before dropping
        bad_channels = list(data[instance].info['bads'])
        
        # Filter out excluded channels if specified
        if excluded_channels:
            channels_to_drop = [ch for ch in bad_channels if ch not in excluded_channels]
            excluded_bads = [ch for ch in bad_channels if ch in excluded_channels]
            if excluded_bads:
                logger.info(f"Excluding {len(excluded_bads)} bad channels from dropping: {excluded_bads}")
        else:
            channels_to_drop = bad_channels
        
        if channels_to_drop:
            # Drop the bad channels
            data[instance].drop_channels(channels_to_drop)
            logger.info(f"Dropped {len(channels_to_drop)} bad channels: {channels_to_drop}")
        else:
            logger.info("No bad channels to drop")

        data['preprocessing_steps'].append({
            'step': 'drop_bad_channels',
            'instance': instance,
            'excluded_channels': excluded_channels,
            'dropped_channels': channels_to_drop,
            'n_bad_channels': len(channels_to_drop)
        })

        return data

    def _step_ica(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ICA for artifact removal."""
        if 'raw' not in data:
            raise ValueError("ica step requires 'raw' in data")

        n_components = step_config.get('n_components', 20)
        random_state = step_config.get('random_state', 97)
        method = step_config.get('method', 'fastica')
        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)

        ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=random_state,
            method=method,
            max_iter='auto'
        )

        # Compute picks if provided
        picks = self._get_picks(data['raw'].info, picks_params, excluded_channels)
        
        ica.fit(data['raw'], picks=picks)

        # Automatically find and exclude artifacts
        excluded_components = defaultdict(list)

        if step_config.get('find_eog', False):
            try:
                eog_indices, eog_scores = ica.find_bads_eog(data['raw'])
                if eog_indices:
                    ica.exclude.extend(eog_indices)
                    for idx in eog_indices:
                        excluded_components[idx].append('eog')
            except Exception:
                # no EOG channels or detection failed
                pass

        if step_config.get('find_ecg', False):
            try:
                ecg_indices, ecg_scores = ica.find_bads_ecg(data['raw'])
                if ecg_indices:
                    ica.exclude.extend(ecg_indices)
                    for idx in ecg_indices:
                        excluded_components[idx].append('ecg')
            except Exception:
                pass
        
        if step_config.get('selected_indices', None):
            selected_indices = step_config.get('selected_indices')
            ica.exclude.extend(selected_indices)
            for idx in selected_indices:
                excluded_components[idx].append('selected')

        # Apply ICA to remove artifacts if requested
        if step_config.get('apply', True):
            ica.apply(data['raw'])

        data['ica'] = ica

        data['preprocessing_steps'].append({
            'step': 'ica',
            'n_components': n_components,
            'method': method,
            'excluded_channels': excluded_channels,
            'excluded_components': len(ica.exclude),
            'component_types': excluded_components,
            'apply': step_config.get('apply', True),
        })

        return data

    def _step_find_events(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Find events in the data."""
        if 'raw' not in data:
            raise ValueError("find_events requires 'raw' in data")

        get_events_from = step_config.get('get_events_from', 'annotations')
        shortest_event = step_config.get('shortest_event', 1)
        event_id = step_config.get('event_id', 'auto')
        
        data['events'], found_event_id = self._find_events_from_raw(
            data['raw'],
            get_events_from=get_events_from,
            shortest_event=shortest_event,
            event_id=event_id
        )
        data['event_id'] = found_event_id
        data['events_sfreq'] = data['raw'].info['sfreq']

        data['preprocessing_steps'].append({
            'step': 'find_events',
            'found_event_id': found_event_id,
            'found_events': data['events'].tolist(),
            'n_events': data['events'].shape[0]
        })

        return data

    def _step_epoch(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create epochs from raw data."""
        if data.get('raw', None) is None or data.get('events', None) is None:
            raise ValueError("epoch step requires both 'raw' and 'events' in data")

        event_id = step_config.get('event_id', None) or data.get('event_id', 'NOT FOUND')
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

    def _step_chunk_in_epoch(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create epochs from raw data."""
        if data.get('raw', None) is None:
            raise ValueError("epoch step requires 'raw' in data")

        duration = step_config.get('duration', 1)

        data['epochs'] = mne.make_fixed_length_epochs(data['raw'], duration=duration, preload=True)

        data['preprocessing_steps'].append({
            'step': 'epoch',
            'type': 'fixed_length_epochs',
            'duration': duration,
        })

        return data

    def _step_find_flat_channels(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find flat channels based on variance threshold.
        
        Flat channels often indicate disconnected electrodes or other hardware issues.
        Channels with variance below the threshold are marked as bad.
        
        Parameters (via step_config)
        -----------------------------
        picks : list, optional
            Channel types to analyze (default: all EEG channels)
        excluded_channels : list, optional
            Channel names to exclude from analysis (e.g., reference channels)
        threshold : float, optional
            Variance threshold below which channels are considered flat
            (default: 1e-12)
        
        Updates
        -------
        data['raw'].info['bads'] : list
            Adds detected flat channels (without duplicates)
        data['preprocessing_steps'] : list
            Appends step information including detected bad channels
        
        Returns
        -------
        data : dict
            Updated data dictionary with flat channels marked as bad
        """
        if 'raw' not in data:
            raise ValueError("find_flat_channels requires 'raw' in data")

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        threshold = step_config.get('threshold', 1e-12)
        
        # Get picks with exclusions
        picks = self._get_picks(data['raw'].info, picks_params, excluded_channels)

        # Get data only for selected picks
        raw_data = data['raw'].get_data(picks=picks)
        variances = raw_data.var(axis=1)
        flat_idx = np.where(variances < threshold)[0]
        # Map back to channel names using picks
        flat_chs = [data['raw'].ch_names[picks[i]] for i in flat_idx]
        
        if flat_chs:
            data['raw'].info['bads'].extend([ch for ch in flat_chs if ch not in data['raw'].info['bads']])
        
        data['preprocessing_steps'].append({
            'step': 'find_flat_channels',
            'instance': 'raw',
            'picks': picks_params,
            'excluded_channels': excluded_channels,
            'apply_on': ['raw'],
            'threshold': threshold,
            'bad_channels': flat_chs,
            'n_bad_channels': len(flat_chs)
        })

        return data

    def _step_find_bads_channels_threshold(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Find bad channels using threshold-based rejection."""
        if 'epochs' not in data:
            raise ValueError("find_bads_channels_threshold requires 'epochs' in data")

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        reject = step_config.get('reject', {'eeg': 100e-6})
        n_epochs_bad_ch = step_config.get('n_epochs_bad_ch', 0.5)
        apply_on = step_config.get('apply_on', ['epochs'])

        if not isinstance(apply_on, list):
            apply_on = [apply_on]

        if any(inst not in data for inst in apply_on):
            raise ValueError(f"find_bads_channels_threshold requires all instances of apply_on ({apply_on}) to be present in data")

        picks = self._get_picks(data['epochs'].info, picks_params, excluded_channels)

        bad_chs = adaptive_reject.find_bads_channels_threshold(
            data['epochs'], picks, reject, n_epochs_bad_ch
        )

        if bad_chs:
            for instance_to_apply in apply_on:
                data[instance_to_apply].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_to_apply].info['bads']])

        data['preprocessing_steps'].append({
            'step': 'find_bads_channels_threshold',
            'picks': picks_params,
            'excluded_channels': excluded_channels,
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
        instance = step_config.get('instance', 'epochs')
        if instance not in data:
            raise ValueError(f"find_bads_channels_variance requires '{instance}' in data")

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        zscore_thresh = step_config.get('zscore_thresh', 4)
        max_iter = step_config.get('max_iter', 2)
        apply_on = step_config.get('apply_on', [instance])

        if not isinstance(apply_on, list):
            apply_on = [apply_on]

        if any(inst not in data for inst in apply_on):
            raise ValueError(f"find_bads_channels_threshold requires all instances of apply_on ({apply_on}) to be present in data")

        picks = self._get_picks(data[instance].info, picks_params, excluded_channels)

        bad_chs = adaptive_reject.find_bads_channels_variance(
            data[instance], picks, zscore_thresh, max_iter
        )

        bad_chs = [data[instance].ch_names[ch_idx] for ch_idx in [1,5,8]]

        # Mark channels as bad
        if bad_chs:
            for instance_to_apply in apply_on:
                data[instance_to_apply].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_to_apply].info['bads']])

        data['preprocessing_steps'].append({
            'step': 'find_bads_channels_variance',
            'instance': instance,
            'picks': picks_params,
            'excluded_channels': excluded_channels,
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
        instance = step_config.get('instance', 'epochs')
        if instance not in data:
            raise ValueError(f"find_bads_channels_high_frequency requires '{instance}' in data")

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        zscore_thresh = step_config.get('zscore_thresh', 4)
        max_iter = step_config.get('max_iter', 2)
        apply_on = step_config.get('apply_on', [instance])

        if not isinstance(apply_on, list):
            apply_on = [apply_on]
        
        if any(inst not in data for inst in apply_on):
            raise ValueError(f"find_bads_channels_threshold requires all instances of apply_on ({apply_on}) to be present in data")

        picks = self._get_picks(data[instance].info, picks_params, excluded_channels)

        bad_chs = adaptive_reject.find_bads_channels_high_frequency(
            data[instance], picks, zscore_thresh, max_iter
        )

        # Mark channels as bad
        if bad_chs:
            for instance_to_apply in apply_on:
                data[instance_to_apply].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_to_apply].info['bads']])

        data['preprocessing_steps'].append({
            'step': 'find_bads_channels_high_frequency',
            'instance': instance,
            'picks': picks_params,
            'excluded_channels': excluded_channels,
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
        excluded_channels = step_config.get('excluded_channels', None)
        reject = step_config.get('reject', {'eeg': 100e-6})
        n_channels_bad_epoch = step_config.get('n_channels_bad_epoch', 0.1)

        picks = self._get_picks(data['epochs'].info, picks_params, excluded_channels)

        bad_epochs = adaptive_reject.find_bads_epochs_threshold(
            data['epochs'], picks, reject, n_channels_bad_epoch
        )

        # Drop bad epochs
        if len(bad_epochs) > 0:
            data['epochs'].drop(bad_epochs, reason='ADAPTIVE AUTOREJECT')

        data['preprocessing_steps'].append({
            'step': 'find_bads_epochs_threshold',
            'picks': picks_params,
            'excluded_channels': excluded_channels,
            'apply_on': ['epochs'], # only for compatibility with others reject steps
            'reject': reject,
            'n_channels_bad_epoch': n_channels_bad_epoch,
            'bad_epochs': bad_epochs.tolist() if hasattr(bad_epochs, 'tolist') else list(bad_epochs),
            'n_bad_epochs': len(bad_epochs),
            'n_epochs_remaining': len(data['epochs'])
        })

        return data

    def _step_save_clean_instance(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Save clean epochs to a BIDS-derivatives-compatible path."""
        instance = step_config.get('instance', 'epochs')
        overwrite = step_config.get('overwrite', True)

        if instance not in data:
            raise ValueError(f"save_clean_instances step requires '{instance}' to be present in data (either 'raw' or 'epochs')")
        
        # Derivatives root for this pipeline
        deriv_root = self.bids_root / "derivatives" / "nice_preprocessing" / instance

        bids_path = BIDSPath(
            subject=data['subject'],
            task=data['task'],
            session=data.get('session', None),
            acquisition=data.get('acquisition', None),
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

        # Save instance
        data[instance].save(bids_path.fpath, overwrite=overwrite)

        # Store paths
        data[f'{instance}_file'] = str(bids_path)

        return data

    def _step_generate_json_report(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON reports."""

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
        import matplotlib.pyplot as plt
        from report import (
            collect_bad_channels_from_steps,
            create_bad_channels_topoplot,
            create_preprocessing_steps_table
        )

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        
        # Get info from epochs if available, otherwise from raw
        info = None
        if 'epochs' in data and data['epochs'] is not None:
            info = data['epochs'].info
        elif 'raw' in data and data['raw'] is not None:
            info = data['raw'].info
        else:
            raise ValueError("generate_html_report requires either 'raw' or 'epochs' in data")
        
        picks = self._get_picks(info, picks_params, excluded_channels)
        plot_raw_kwargs = step_config.get('plot_raw_kwargs', {})
        plot_ica_kwargs = step_config.get('plot_ica_kwargs', {})
        plot_events_kwargs = step_config.get('plot_events_kwargs', {})
        plot_epochs_kwargs = step_config.get('plot_epochs_kwargs', {})
        plot_evokeds_kwargs = step_config.get('plot_evokeds_kwargs', {})

        raw = data.get('raw', None)
        if raw is not None:
            raw = raw.copy().pick(picks=picks, exclude='bads')
        
        epochs = data.get('epochs', None)
        if epochs is not None:
            epochs = epochs.copy().pick(picks=picks, exclude='bads')

        html_report = mne.Report(title=f'Preprocessing Report - Subject {data["subject"]}')

        # Add bad channels topoplot section
        try:
            # Collect bad channels from all preprocessing steps
            bad_channels = collect_bad_channels_from_steps(
                data.get('preprocessing_steps', [])
            )
            
            # Get info from raw to use its montage
            info = None
            if 'raw' in data and data['raw'] is not None:
                info = data['raw'].info
            elif 'epochs' in data and data['epochs'] is not None:
                info = data['epochs'].info
            
            # Create topoplot if we have bad channels and info
            if info is not None and bad_channels:
                fig = create_bad_channels_topoplot(info, bad_channels)
                
                if fig is not None:
                    # Add to report
                    html_report.add_figure(
                        fig=fig,
                        title='Bad Channels',
                        caption=f'Topoplot showing {len(bad_channels)} bad channels marked with red crosses'
                    )
                    plt.close(fig)
        except Exception as e:
            # If adding bad channels topoplot fails, continue without stopping the pipeline
            logger.warning(f"Failed to add bad channels topoplot: {e}")
            pass

        # Add preprocessing steps table section
        try:
            if 'preprocessing_steps' in data and len(data['preprocessing_steps']) > 0:
                # Create HTML table with collapsible rows
                html_content = create_preprocessing_steps_table(data['preprocessing_steps'])
                
                if html_content:
                    # Add the HTML table to the report
                    html_report.add_html(
                        html=html_content,
                        title='Preprocessing Steps',
                    )
        except Exception as e:
            # If adding preprocessing steps table fails, continue without stopping the pipeline
            logger.warning(f"Failed to add preprocessing steps table: {e}")
            pass

        if data.get('ica', None) is not None and raw is not None:
            try:
                html_report.add_ica(
                    ica=data['ica'],
                    title='ICA Components',
                    inst=raw,
                    **plot_ica_kwargs
                )
            except Exception:
                # If adding ICA fails, continue without stopping the pipeline
                pass

        if raw is not None:
            try:
                html_report.add_raw(
                    raw=raw,
                    title='Clean Raw Data',
                    **plot_raw_kwargs
                )
            except Exception:
                pass

        if 'events' in data and data['events'] is not None:
            try:
                html_report.add_events(
                    events=data['events'],
                    event_id=data.get('event_id', None),
                    sfreq=data['events_sfreq'],
                    title='Found Events',
                    **plot_events_kwargs
                )
            except Exception:
                pass

        if 'epochs' in data and data['epochs'] is not None:

            try:

                html_report.add_epochs(
                    epochs=epochs,
                    title='Clean Epochs',
                    **plot_epochs_kwargs
                )
                
                html_report.add_evokeds(
                    evokeds=epochs.average(by_event_type=True),
                    n_time_points=step_config.get('n_time_points', None),
                    **plot_evokeds_kwargs
                )
            except Exception:
                pass

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

        data['html_report'] = str(bids_path)
        return data

    def _process_single_recording(self, bids_path: list[BIDSPath], progress: Progress = None, task_id: int = None) -> Dict[str, Any]:
        """Process a single subject using the configured pipeline steps."""
        # Initialize data dictionary
        data = {
            'subject': bids_path[0].subject,
            'task': bids_path[0].task,
            'session': bids_path[0].session,
            'acquisition': bids_path[0].acquisition,
            'preprocessing_steps': []
        }

        # Read BIDS data
        logger.info(f"Reading BIDS data from:")
        for bp in bids_path:
            logger.info(f"  - {bp.fpath}")

        # Read all BIDS raws and concatenate into a single Raw object
        data['all_raw'] = [read_raw_bids(bids_path=bp, verbose=False) for bp in bids_path]

        # Ensure data are loaded into memory for processing
        for raw in data['all_raw']:
            raw.load_data()

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
            'raw_files': [str(bp.fpath) for bp in bids_path],
        }

        # Copy relevant output information to results
        for key in ['raw_file', 'epochs_file', 'json_report', 'html_report', 'n_epochs', 'preprocessing_steps']:
            if key in data:
                results[key] = data[key]

        logger.info(f"Successfully processed {data.get('subject')} - {data.get('session')} - {data.get('task')} - {data.get('acquisition')}")
        return results

    def run_pipeline(
        self, 
        subjects: Union[str, List[str]] = None,
        sessions: Union[str, List[str]] = None,
        tasks: Union[str, List[str]] = None,
        acquisitions: Union[str, List[str]] = None,
        extension: str = '.vhdr'
    ) -> Dict[str, Any]:
        """Run the pipeline using mne-bids to query files.

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

        Returns
        -------
        all_results : dict
            Dictionary mapping subject -> list of results for each matching file.
        """
        
        subjects = self._get_entity_values('subject', subjects)
        sessions = self._get_entity_values('session', sessions)
        tasks = self._get_entity_values('task', tasks)
        acquisitions = self._get_entity_values('acquisition', acquisitions)
        
        # print subjects, sessions, tasks, acquisitions
        logger.info(f"Subjects to process: {subjects}")
        logger.info(f"Sessions to process: {sessions}")
        logger.info(f"Tasks to process: {tasks}")
        logger.info(f"Acquisitions to process: {acquisitions}")

        n_combinations = len(subjects) * len(sessions) * len(tasks) * len(acquisitions)
        logger.info(f"Computing {n_combinations} matching file(s) to process")

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
                total=n_combinations
            )

            for i, (subject, session, task, acquisition) in enumerate(product(subjects, sessions, tasks, acquisitions)):

                pb = BIDSPath(
                    root=self.bids_root,
                    subject=subject,
                    session=session,
                    task=task,
                    acquisition=acquisition,
                    extension=extension,
                    suffix='eeg',
                    datatype='eeg',
                )

                all_raw_paths = list(pb.match(ignore_nosub=True))
                logger.info(f"Found {len(all_raw_paths)} recording(s) for {subject} - {session} - {task} - {acquisition} to process together.")

                # Get pipeline steps for this recording's progress bar
                pipeline_steps = self._get_pipeline_steps()
                
                # Create a task for the current recording's steps
                recording_name = f"{subject} - {session} - {task} - {acquisition}"
                step_task = progress.add_task(
                    f"[cyan]{recording_name}", 
                    total=len(pipeline_steps)
                )
                
                try:
                    results = self._process_single_recording(all_raw_paths, progress, step_task)
                    all_results.setdefault(subject, []).append(results)
                    logger.info(f"Successfully completed {recording_name}")
                except Exception as exc:
                    # Do not stop the whole batch if one subject fails; capture the error
                    logger.error(f"Error processing {recording_name}: {str(exc)}")
                    all_results.setdefault(subject, []).append({'error': str(exc)})
                    raise exc
                finally:
                    # Remove the step task after this recording is done
                    progress.remove_task(step_task)
                
                # Update overall progress
                progress.update(overall_task, completed=i+1)

        logger.info(f"Pipeline completed.")
        return all_results