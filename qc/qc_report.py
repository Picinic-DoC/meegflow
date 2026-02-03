# qc_report.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from qc_metrics import (
    QCThresholds,
    _pick_eeg_names,
    oha,
    thv,
    chv,
    rbc,
    roi_numbers_to_names,
)
from qc_classify import ROIQCConfig, pass_fail_roi
from qc_mad import compute_thresholds_from_df, classify_good_bad


@dataclass
class PathsConfig:
    base_epochs: Path
    out_dir: Path
    task: str = "FCSRT"
    sessions: Tuple[str, ...] = ("ses-M0", "ses-M12", "ses-M24", "ses-M36", "ses-M48", "ses-M60")


def list_subjects(base: Path) -> List[str]:
    return sorted([p.name for p in base.glob("sub-*") if p.is_dir()])


def find_epochs_file(base_epochs: Path, sub: str, ses: str, task: str) -> Path:
    return base_epochs / sub / ses / "eeg" / f"{sub}_{ses}_task-{task}_proc-clean_desc-cleaned_epo.fif"


def plot_hist_with_thr(df: pd.DataFrame, col: str, thr: float, out_png: Path, title: str) -> None:
    fig = plt.figure()
    x = df[col].dropna().values
    plt.hist(x, bins=50)
    plt.axvline(thr, linewidth=2)   # default color
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("count")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_scatter_oha_chv(df: pd.DataFrame, thr_oha: float, thr_chv: float, out_png: Path) -> None:
    fig = plt.figure()
    x = df["global_oha"].values
    y = df["global_chv"].values

    if "global_label_mad" in df.columns:
        labels = df["global_label_mad"].values
        mask_bad = labels == "BAD"
        plt.scatter(x[~mask_bad], y[~mask_bad], label="GOOD")
        plt.scatter(x[mask_bad], y[mask_bad], label="BAD")
        plt.legend()
    else:
        plt.scatter(x, y)

    plt.axvline(thr_oha, linewidth=2)
    plt.axhline(thr_chv, linewidth=2)
    plt.xlabel("global_oha")
    plt.ylabel("global_chv")
    plt.title("OHA vs CHV (MAD thresholds)")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def write_html_report(
    df: pd.DataFrame,
    thresholds: dict,
    out_dir: Path,
    task: str,
    mad_k: float,
) -> None:
    html_path = out_dir / f"qc_report_task-{task}.html"
    thr_df = pd.DataFrame([thresholds])

    html = f"""
    <html>
    <head>
        <title>QC report – {task}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ccc; padding: 4px; font-size: 12px; }}
            th {{ background-color: #eee; }}
        </style>
    </head>
    <body>

    <h1>QC report – task {task}</h1>

    <h2>Thresholds (median + {mad_k}*MAD)</h2>
    {thr_df.to_html(index=False)}

    <h2>Distributions</h2>
    <h3>Global OHA</h3>
    <img src="hist_global_oha_mad.png" width="600">

    <h3>Global THV</h3>
    <img src="hist_global_thv_mad.png" width="600">

    <h3>Global CHV</h3>
    <img src="hist_global_chv_mad.png" width="600">

    <h2>OHA vs CHV</h2>
    <img src="scatter_oha_vs_chv_mad.png" width="600">

    <h2>QC table</h2>
    {df.to_html(index=False)}

    </body>
    </html>
    """

    with open(html_path, "w") as f:
        f.write(html)

    print(f"[SAVED] {html_path}")


def run_qc_mad(
    paths: PathsConfig,
    thresholds_uv: QCThresholds,
    rois_numbers: Dict[str, Sequence[int]],
    roi_cfgs: Dict[str, ROIQCConfig],
    roi_prefix: str = "E",
    mad_k: float = 3.0,
    expected_n_channels: int = 256,
    mne_log_level: str = "WARNING",
) -> pd.DataFrame:
    """
    Fast version:
    - read epochs header (preload=False)
    - load data ONCE per recording (global picks)
    - compute ROI metrics by slicing (no extra reads)
    """
    mne.set_log_level(mne_log_level)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    subjects = list_subjects(paths.base_epochs)
    if not subjects:
        raise RuntimeError(f"No subjects found under {paths.base_epochs}")

    rows: List[Dict[str, object]] = []

    for ses in paths.sessions:
        for sub in subjects:
            epo_fif = find_epochs_file(paths.base_epochs, sub, ses, paths.task)
            if not epo_fif.exists():
                continue

            print(f"[QC] {ses} {sub} ...", flush=True)

            epochs = mne.read_epochs(epo_fif, preload=False)

            # --- 1) définir picks EEG (sans bads) une seule fois
            eeg_names = _pick_eeg_names(epochs.info)
            bads = set(epochs.info.get("bads", []))

            global_ch_names = [ch for ch in eeg_names if ch in epochs.ch_names and ch not in bads]
            global_picks = mne.pick_channels(epochs.ch_names, include=global_ch_names, exclude=[])

            # --- 2) lire les données UNE seule fois
            if len(global_picks) == 0:
                # Rien à lire -> metrics NaN (évite warnings numpy)
                data_uV_global = None
            else:
                data_uV_global = epochs.get_data(picks=global_picks) * 1e6  # (n_epochs, n_ch, n_times)

            # --- 3) global metrics depuis l’array déjà en mémoire
            if data_uV_global is None or data_uV_global.shape[1] == 0:
                global_oha = float("nan")
                global_thv = float("nan")
                global_chv = float("nan")
            else:
                global_oha = oha(data_uV_global, thresholds_uv.oha_uv)
                global_thv = thv(data_uV_global, thresholds_uv.hv_uv)
                global_chv = chv(data_uV_global, thresholds_uv.hv_uv)

            n_total, n_bad, ratio = rbc(epochs.info, roi_ch_names=None)

            # bookkeeping
            n_present = len([ch for ch in epochs.ch_names if ch.startswith("E")])
            n_missing = max(expected_n_channels - n_present, 0)
            n_bads = len(epochs.info.get("bads", []))

            row: Dict[str, object] = {
                "subject": sub,
                "session": ses,
                "epochs_file": str(epo_fif),
                "global_oha": global_oha,
                "global_thv": global_thv,
                "global_chv": global_chv,
                "global_rbc": float(ratio),
                "global_n_total": int(n_total),
                "global_n_bad": int(n_bad),
                "n_channels_present": int(n_present),
                f"n_channels_missing_vs_{expected_n_channels}": int(n_missing),
                "n_bads_listed": int(n_bads),
            }

            # --- 4) ROI metrics par slicing (zéro relecture disque)
            name_to_local_idx = {ch: i for i, ch in enumerate(global_ch_names)}

            for roi_name, roi_nums in rois_numbers.items():
                roi_ch = roi_numbers_to_names(epochs.ch_names, roi_nums, prefix=roi_prefix)
                roi_ch = [ch for ch in roi_ch if ch in name_to_local_idx]

                if data_uV_global is None or len(roi_ch) == 0:
                    roi_oha = float("nan")
                    roi_thv = float("nan")
                    roi_chv = float("nan")
                    roi_n_total, roi_n_bad, roi_rbc = (0, 0, float("nan"))
                else:
                    roi_local_idx = [name_to_local_idx[ch] for ch in roi_ch]
                    data_uV_roi = data_uV_global[:, roi_local_idx, :]

                    roi_oha = oha(data_uV_roi, thresholds_uv.oha_uv)
                    roi_thv = thv(data_uV_roi, thresholds_uv.hv_uv)
                    roi_chv = chv(data_uV_roi, thresholds_uv.hv_uv)

                    roi_n_total, roi_n_bad, roi_rbc = rbc(epochs.info, roi_ch_names=roi_ch)

                row[f"{roi_name}_oha"] = roi_oha
                row[f"{roi_name}_thv"] = roi_thv
                row[f"{roi_name}_chv"] = roi_chv
                row[f"{roi_name}_rbc"] = float(roi_rbc)
                row[f"{roi_name}_n_total"] = int(roi_n_total)
                row[f"{roi_name}_n_bad"] = int(roi_n_bad)

                cfg = roi_cfgs.get(roi_name, None)
                if cfg is None:
                    row[f"{roi_name}_pass"] = None
                    row[f"{roi_name}_reason"] = "NO_ROI_CFG"
                else:
                    ok, reason = pass_fail_roi(
                        {"n_total": float(roi_n_total), "n_bad": float(roi_n_bad), "rbc": float(roi_rbc)},
                        cfg
                    )
                    row[f"{roi_name}_pass"] = bool(ok)
                    row[f"{roi_name}_reason"] = reason

            rows.append(row)

            # checkpoint
            if len(rows) % 50 == 0:
                partial_path = paths.out_dir / f"qc_metrics_task-{paths.task}_PARTIAL.csv"
                pd.DataFrame(rows).to_csv(partial_path, index=False)
                print(f"[CHECKPOINT] wrote {len(rows)} rows -> {partial_path.name}", flush=True)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        out_csv = paths.out_dir / f"qc_metrics_task-{paths.task}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[SAVED EMPTY] {out_csv}")
        return df

    # 3) MAD thresholds (global)
    thr = compute_thresholds_from_df(df, k=mad_k)
    df["global_label_mad"] = classify_good_bad(df, thr)

    thr_csv = paths.out_dir / f"qc_thresholds_mad_k{mad_k:g}_task-{paths.task}.csv"
    pd.DataFrame([thr]).to_csv(thr_csv, index=False)
    print(f"[SAVED] {thr_csv}")

    out_csv = paths.out_dir / f"qc_metrics_task-{paths.task}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    # 5) Plots
    plot_hist_with_thr(df, "global_oha", thr["global_oha_thr"], paths.out_dir / "hist_global_oha_mad.png",
                       f"Global OHA (thr = median + {mad_k}*MAD)")
    plot_hist_with_thr(df, "global_thv", thr["global_thv_thr"], paths.out_dir / "hist_global_thv_mad.png",
                       f"Global THV (thr = median + {mad_k}*MAD)")
    plot_hist_with_thr(df, "global_chv", thr["global_chv_thr"], paths.out_dir / "hist_global_chv_mad.png",
                       f"Global CHV (thr = median + {mad_k}*MAD)")

    plot_scatter_oha_chv(df, thr["global_oha_thr"], thr["global_chv_thr"],
                         paths.out_dir / "scatter_oha_vs_chv_mad.png")

    # 6) HTML
    write_html_report(df=df, thresholds=thr, out_dir=paths.out_dir, task=paths.task, mad_k=mad_k)

    return df
