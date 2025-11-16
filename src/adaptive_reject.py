# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import math
import numpy as np

from scipy.signal import butter, filtfilt
from scipy.stats import zscore

import mne
from mne.utils import logger


def _iteratively_find_outliers(X, threshold=3.0, max_iter=4):
    """Find outliers based on iterated Z-scoring. """

    X = np.ma.masked_array(X, mask=False)
    for _ in range(max_iter):
        
        X_z = np.abs(zscore(X))
        current_bad = X_z > threshold
        if np.all(~current_bad):
            break
        
        X.mask |= current_bad

    return np.where(X.mask)[0]


def find_bads_channels_threshold(epochs, picks, reject, n_epochs_bad_ch=0.5):

    n_channels = len(picks)
    data = epochs.get_data()
    n_epochs = data.shape[0]

    if isinstance(n_epochs_bad_ch, float):
        n_epochs_bad_ch = math.floor(n_epochs_bad_ch * n_epochs)

    ch_types_inds = mne.channel_indices_by_type(epochs.info)
    data = np.transpose(data, (1, 0, 2))
    bad_ch_idx = np.ndarray((0,), dtype=int)
    for key, reject_thresh in reject.items():
        idx = np.array([x for x in ch_types_inds[key] if x in picks])
        count_bad_epochs = np.zeros((n_channels), dtype=int)
        for i_ch, channel in enumerate(data[idx]):
            deltas = channel.max(axis=1) - channel.min(axis=1)
            idx_deltas = np.where(np.greater(deltas, reject_thresh))[0]
            count_bad_epochs[i_ch] = idx_deltas.shape[0]
        reject_bad_channels = np.where(count_bad_epochs > n_epochs_bad_ch)[0]
        logger.info('Reject by threshold %f on %s %d : bad_channels: %s' %
                    (reject_thresh, key.upper(), len(reject_bad_channels),
                     reject_bad_channels))
        bad_ch_idx = np.concatenate((bad_ch_idx, reject_bad_channels))

    bad_chs = list({epochs.ch_names[i] for i in bad_ch_idx})
    return bad_chs


def find_bads_channels_variance(inst, picks, zscore_thresh=4, max_iter=2):
    logger.info('Looking for bad channels with variance')
    if isinstance(inst, mne.Epochs):
        data = inst.get_data()
    else:
        data = inst.get_data()[None, :]
    masked_data = np.ma.masked_array(data, fill_value=np.nan)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    ch_var = np.ma.hstack(masked_data).var(axis=-1)
    bad_ch_var = _iteratively_find_outliers(
        ch_var, threshold=zscore_thresh, max_iter=max_iter)
    logger.info('Reject by variance: bad_channels: %s' % bad_ch_var)
    bad_chs = list({inst.ch_names[i] for i in bad_ch_var})
    return bad_chs


def find_bads_channels_high_frequency(inst, picks, zscore_thresh=4, max_iter=2):
    logger.info('Looking for bad channels with high frequency variance')
    if isinstance(inst, mne.Epochs):
        data = inst.get_data()
    else:
        data = inst.get_data()[None, :]
    masked_data = np.ma.masked_array(data, fill_value=np.nan)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    filter_freq = 25
    b, a = butter(4, 2.0 * filter_freq / inst.info['sfreq'], 'highpass')
    filt_data = filtfilt(b, a, np.ma.hstack(masked_data))
    filt_masked_data = np.ma.masked_array(filt_data, fill_value=np.nan)
    if len(exclude) > 0:
        filt_masked_data[exclude, :] = np.ma.masked
    bad_ch_hf = _iteratively_find_outliers(
        filt_masked_data.std(axis=-1), threshold=zscore_thresh,
        max_iter=max_iter)
    logger.info('Reject by high frequency std: bad_channels: %s' % bad_ch_hf)
    bad_chs = list({inst.ch_names[i] for i in bad_ch_hf})
    return bad_chs


def find_bads_epochs_threshold(epochs, picks, reject, n_channels_bad_epoch=0.1):
    n_channels = len(picks)
    bad_ep_idx = np.ndarray((0,), dtype=int)
    if isinstance(n_channels_bad_epoch, float):
        n_channels_bad_epoch = math.floor(n_channels_bad_epoch * n_channels)

    data = epochs.get_data()
    masked_data = np.ma.masked_array(data, fill_value=np.nan)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    ch_types_inds = mne.channel_indices_by_type(epochs.info)
    n_epochs = masked_data.shape[0]
    for key, reject_thresh in reject.items():
        idx = np.array([x for x in ch_types_inds[key] if x in picks])
        count_bad_chans = np.zeros((n_epochs), dtype=int)
        for i_ep, epoch in enumerate(masked_data[:, idx]):
            deltas = epoch.max(axis=1) - epoch.min(axis=1)
            idx_deltas = np.where(np.greater(deltas, reject_thresh))[0]
            count_bad_chans[i_ep] = idx_deltas.shape[0]
        reject_bad_epochs = np.where(count_bad_chans > n_channels_bad_epoch)[0]
        logger.info('Reject by threshold %f on %s : bad_epochs: %s' %
                    (reject_thresh, key.upper(), reject_bad_epochs))
    bad_epochs = np.unique(np.concatenate((bad_ep_idx, reject_bad_epochs)))

    return bad_epochs

