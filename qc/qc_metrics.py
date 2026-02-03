# qc_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import mne


@dataclass
class QCThresholds:
    # thresholds in microvolts
    oha_uv: float = 50.0
    hv_uv: float = 15.0


def _pick_eeg_names(info: mne.Info) -> List[str]:
    picks = mne.pick_types(info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
    return [info["ch_names"][i] for i in picks]


def _epochs_data_uV(epochs: mne.Epochs, ch_names: Optional[Sequence[str]] = None) -> np.ndarray:
    """
    Returns data in microvolts: shape (n_epochs, n_channels, n_times).
    Excludes channels marked as bad (signal final post-preprocessing).
    """
    bads = epochs.info.get("bads", [])

    if ch_names is None:
        eeg_names = _pick_eeg_names(epochs.info)
        picks = mne.pick_channels(epochs.ch_names, include=eeg_names, exclude=bads)
    else:
        eeg_set = set(_pick_eeg_names(epochs.info))
        roi_keep = [ch for ch in ch_names if ch in eeg_set]
        picks = mne.pick_channels(epochs.ch_names, include=roi_keep, exclude=bads)

    if len(picks) == 0:
        return np.empty((len(epochs), 0, len(epochs.times)), dtype=float)

    data_V = epochs.get_data(picks=picks)  # volts
    return data_V * 1e6  # microvolts


def oha(data_uV: np.ndarray, thr_uV: float) -> float:
    """Overall High Amplitude: proportion of points with |x| > thr."""
    return float(np.mean(np.abs(data_uV) > thr_uV))


def thv(data_uV: np.ndarray, thr_uV: float) -> float:
    """
    Timepoints of High Variance:
    For each epoch and timepoint compute SD across channels;
    ratio of (epoch,time) with SD > thr.
    """
    sd_across_ch = np.std(data_uV, axis=1, ddof=0)  # (n_epochs, n_times)
    return float(np.mean(sd_across_ch > thr_uV))


def chv(data_uV: np.ndarray, thr_uV: float) -> float:
    """
    Channels of High Variance:
    SD across time per epoch/channel, then mean across epochs per channel,
    ratio of channels whose mean SD > thr.
    """
    sd_across_time = np.std(data_uV, axis=2, ddof=0)  # (n_epochs, n_channels)
    mean_sd_per_ch = np.mean(sd_across_time, axis=0)  # (n_channels,)
    return float(np.mean(mean_sd_per_ch > thr_uV))


def rbc(info: mne.Info, roi_ch_names: Optional[Sequence[str]] = None) -> Tuple[int, int, float]:
    """
    Ratio of bad channels (channels marked in info['bads']) among EEG channels.
    If roi_ch_names provided, computed within ROI only.
    Returns: (n_total, n_bad, ratio)
    """
    bads = set(info.get("bads", []))
    eeg_names = set(_pick_eeg_names(info))

    if roi_ch_names is None:
        chs = sorted(list(eeg_names))
    else:
        chs = [ch for ch in roi_ch_names if ch in eeg_names]

    n_total = len(chs)
    n_bad = sum(1 for ch in chs if ch in bads)
    ratio = (n_bad / n_total) if n_total > 0 else float("nan")
    return n_total, n_bad, float(ratio)


def roi_numbers_to_names(ch_names: Sequence[str], roi_numbers: Sequence[int], prefix: str = "E") -> List[str]:
    """
    Convert numeric ROI labels (e.g., 13, 224) to channel names (e.g., E13, E224),
    keeping only those that exist in ch_names.
    """
    wanted = [f"{prefix}{n}" for n in roi_numbers]
    present = set(ch_names)
    return [ch for ch in wanted if ch in present]


def compute_metrics_epochs(
    epochs: mne.Epochs,
    thresholds: QCThresholds,
    roi_ch_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Compute OHA/THV/CHV on epochs, either global EEG or restricted to ROI channels.
    RBC is computed from info['bads'] (global or ROI).
    """
    data_uV = _epochs_data_uV(epochs, ch_names=roi_ch_names)

    out: Dict[str, float] = {}

# ---- guard: 0 channels after excluding bads (or missing ROI)
    if data_uV.shape[1] == 0:
        out["oha"] = float("nan")
        out["thv"] = float("nan")
        out["chv"] = float("nan")
    else:
        out["oha"] = oha(data_uV, thresholds.oha_uv)
        out["thv"] = thv(data_uV, thresholds.hv_uv)
        out["chv"] = chv(data_uV, thresholds.hv_uv)


    n_total, n_bad, ratio = rbc(epochs.info, roi_ch_names=roi_ch_names)
    out["n_total"] = float(n_total)
    out["n_bad"] = float(n_bad)
    out["rbc"] = float(ratio)
    return out
