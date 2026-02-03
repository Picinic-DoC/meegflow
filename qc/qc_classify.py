# qc_classify.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


@dataclass
class GlobalQCConfig:
    # thresholds apply to metrics computed globally
    # You can tune these; defaults are conservative-ish.
    good_max_oha: float = 0.01
    ok_max_oha: float = 0.05

    good_max_thv: float = 0.01
    ok_max_thv: float = 0.05

    good_max_chv: float = 0.10
    ok_max_chv: float = 0.25

    good_max_rbc: float = 0.10
    ok_max_rbc: float = 0.25


@dataclass
class ROIQCConfig:
    # You said: "define my thresholds for the 4"
    # Provide per-ROI thresholds below (example structure).
    # Interpretation: PASS if rbc <= max_rbc and n_good >= min_good
    max_rbc: float
    min_good: int


def classify_global(metrics: Dict[str, float], cfg: GlobalQCConfig) -> str:
    """
    Simple rule-based Good/OK/Bad.
    You can refine weights, but keep it transparent.
    """
    oha = metrics["oha"]
    thv = metrics["thv"]
    chv = metrics["chv"]
    rbc = metrics["rbc"]

    if (oha <= cfg.good_max_oha and thv <= cfg.good_max_thv and
        chv <= cfg.good_max_chv and rbc <= cfg.good_max_rbc):
        return "GOOD"

    if (oha <= cfg.ok_max_oha and thv <= cfg.ok_max_thv and
        chv <= cfg.ok_max_chv and rbc <= cfg.ok_max_rbc):
        return "OK"

    return "BAD"


def pass_fail_roi(metrics_roi: Dict[str, float], roi_cfg: ROIQCConfig) -> Tuple[bool, str]:
    """
    ROI decision for a given effect/ROI.
    Uses:
      - rbc (within ROI, computed from info['bads'])
      - n_total / n_bad to get n_good
    """
    n_total = int(metrics_roi["n_total"])
    n_bad = int(metrics_roi["n_bad"])
    rbc = metrics_roi["rbc"]
    n_good = n_total - n_bad

    if n_total == 0:
        return False, "ROI_NO_CHANNELS"

    if rbc > roi_cfg.max_rbc:
        return False, f"ROI_RBC_TOO_HIGH(rbc={rbc:.3f} > {roi_cfg.max_rbc:.3f})"

    if n_good < roi_cfg.min_good:
        return False, f"ROI_TOO_FEW_GOOD(n_good={n_good} < {roi_cfg.min_good})"

    return True, "PASS"
