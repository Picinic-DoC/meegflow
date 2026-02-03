# qc_mad.py
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

def mad(x: np.ndarray) -> float:
    """
    Median Absolute Deviation (MAD): median(|x - median(x)|)
    Robust dispersion estimator.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def robust_threshold(x: np.ndarray, k: float = 3.0) -> Tuple[float, float, float]:
    """
    Returns (median, MAD, threshold = median + k*MAD)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(x))
    m = mad(x)
    thr = med + k * m
    return med, m, float(thr)

def compute_thresholds_from_df(df, k: float = 3.0) -> Dict[str, float]:
    """
    Expects columns: global_oha, global_thv, global_chv
    Returns thresholds dict.
    """
    thr: Dict[str, float] = {}
    for col in ["global_oha", "global_thv", "global_chv"]:
        med, m, t = robust_threshold(df[col].values, k=k)
        thr[f"{col}_median"] = med
        thr[f"{col}_mad"] = m
        thr[f"{col}_thr"] = t
    return thr

def classify_good_bad(df, thresholds: Dict[str, float]) -> np.ndarray:
    """
    BAD if any metric exceeds its MAD-threshold.
    """
    oha_bad = df["global_oha"] > thresholds["global_oha_thr"]
    thv_bad = df["global_thv"] > thresholds["global_thv_thr"]
    chv_bad = df["global_chv"] > thresholds["global_chv_thr"]
    bad = (oha_bad | thv_bad | chv_bad)
    return np.where(bad, "BAD", "GOOD")
