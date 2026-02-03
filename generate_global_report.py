#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate GROUP HTML reports (one per session) for NICE preprocessing outputs.

Per session report includes:
1) ICA excluded component topographies (only excluded comps) - loaded from PNG files saved by nice_preprocessing
2) ICA EOG scores barplots (per subject) - from JSON
3) Time course EEG: ERP per subject averaged across 3 conditions - from epochs FIF

Inputs:
- Individual reports (JSON + optional HTML):
  {reports_root}/sub-XXX/ses-<SESSION>/eeg/*_report.json
  {reports_root}/sub-XXX/ses-<SESSION>/eeg/*_report.html
- Preprocessed epochs:
  {epochs_root}/sub-XXX/ses-<SESSION>/eeg/*.fif

ICA topographies PNGs (preferred):
  {bids_root}/derivatives/nice_preprocessing/ica_topos/sub-XXX/ses-<SESSION>/eeg/sub-XXX_ses-<SESSION>_ica_comp-000.png

Dependencies:
- mne
- numpy
- matplotlib
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

import mne


# ----------------------------
# Defaults
# ----------------------------

DEFAULT_CONDITIONS = [
    "Stimulus/CatNewRepeated/CR",
    "Stimulus/CatNewUnique/CR",
    "Stimulus/CatOld/Hit",
]

DEFAULT_SESSIONS = ["M0", "M12", "M24", "M36", "M48", "M60"]


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class SubjectSessionPaths:
    subject: str          # e.g. "023"
    session: str          # e.g. "M0"
    json_path: Path
    html_path: Optional[Path]
    epochs_fif: Optional[Path]


# ----------------------------
# Utility helpers
# ----------------------------

def _safe_int_subject(sub: str) -> int:
    try:
        return int(sub)
    except Exception:
        return 10**9


def _b64_from_matplotlib_fig(fig) -> str:
    """Return a PNG data-URI from a matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_ica_step(report: Dict) -> Optional[Dict]:
    steps = report.get("preprocessing_steps", [])
    if not isinstance(steps, list):
        return None
    ica_steps = [s for s in steps if isinstance(s, dict) and s.get("step") == "ica"]
    return ica_steps[-1] if ica_steps else None


def _extract_ica_excluded_and_eog(report: Dict) -> Tuple[List[int], List[int], Optional[np.ndarray], List[str]]:
    """
    Returns:
      excluded_components (global)
      eog_excluded_components
      eog_scores (np array or None)
      eog_channels_present (list)
    """
    ica_step = _find_latest_ica_step(report) or {}
    excluded = ica_step.get("excluded_components", []) or []
    excluded = [int(x) for x in excluded] if isinstance(excluded, list) else []

    eog = ica_step.get("eog_detection", {}) or {}
    eog_excl = eog.get("eog_excluded_components", []) or []
    eog_excl = [int(x) for x in eog_excl] if isinstance(eog_excl, list) else []

    eog_scores = eog.get("eog_scores", None)
    eog_scores_arr = None
    if eog_scores is not None:
        try:
            eog_scores_arr = np.array(eog_scores, dtype=float)
        except Exception:
            eog_scores_arr = None

    eog_ch = eog.get("eog_channels_present", None) or eog.get("eog_channels_requested", None) or []
    if not isinstance(eog_ch, list):
        eog_ch = []

    return excluded, eog_excl, eog_scores_arr, eog_ch


def _plot_eog_scores_barplot(eog_scores: np.ndarray, eog_excluded: Sequence[int], title_suffix: str = ""):
    """
    Mimics your individual-report logic:
    - If scores is 2D: aggregate to 1 score per component via max(abs(scores), axis=0)
    - If 1D: abs(scores)
    """
    if eog_scores.ndim == 1:
        scores_1d = np.abs(eog_scores)
        ylabel = "|EOG score|"
    else:
        scores_1d = np.max(np.abs(eog_scores), axis=0)
        ylabel = "max |EOG score| across EOG channels"

    fig = plt.figure(figsize=(9.5, 3.3))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(scores_1d)), scores_1d)
    ax.set_xlabel("ICA component")
    ax.set_ylabel(ylabel)
    ax.set_title(f"EOG scores (selected: {list(eog_excluded)}) {title_suffix}".strip())
    fig.tight_layout()
    return fig


def _find_epochs_fif(epochs_dir: Path) -> Optional[Path]:
    if not epochs_dir.exists():
        return None

    patterns = [
        "*epo.fif",
        "*epo.fif.gz",
        "*epochs*.fif",
        "*epochs*.fif.gz",
        "*.fif",
        "*.fif.gz",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(sorted(epochs_dir.glob(pat)))

    filtered = []
    for p in candidates:
        name = p.name.lower()
        if "raw" in name:
            continue
        if "report" in name:
            continue
        if "ica" in name:
            continue
        filtered.append(p)

    return filtered[0] if filtered else None


def _read_epochs(path: Path) -> mne.Epochs:
    return mne.read_epochs(path, preload=False, verbose="ERROR")


def _compute_subject_erp_plot(
    epochs_path: Path,
    conditions: Sequence[str],
    title: str,
    n_topomaps: int = 3,
) -> Optional[plt.Figure]:
    """
    ERP QC figure per subject:
    - average each condition
    - combine equally across conditions
    - plot joint figure: butterfly (spatial_colors) + topomaps at selected times
    """
    try:
        epochs = _read_epochs(epochs_path)
    except Exception:
        return None

    # EEG only
    try:
        epochs = epochs.copy().pick("eeg")
    except Exception:
        pass

    if not isinstance(epochs.event_id, dict):
        return None

    available = set(epochs.event_id.keys())
    conds_present = [c for c in conditions if c in available]
    if len(conds_present) == 0:
        return None

    evokeds = []
    for c in conds_present:
        try:
            evokeds.append(epochs[c].average())
        except Exception:
            continue
    if len(evokeds) == 0:
        return None

    # Equal average across conditions
    try:
        evoked = mne.combine_evoked(evokeds, weights="equal")
    except Exception:
        data = np.mean([e.data for e in evokeds], axis=0)
        evoked = evokeds[0].copy()
        evoked.data = data

        # ---- Option A: keep equal averaging across conditions, but display an integer N ----
    # naves are integers (number of epochs per condition)
    n_by_cond = {c: int(ev.nave) for c, ev in zip(conds_present, evokeds)}
    n_total = int(sum(n_by_cond.values()))

    # MNE may create a non-integer "equivalent nave" after combine_evoked.
    # For QC display, we override nave to a meaningful integer.
    evoked.nave = n_total

    # (Optional but recommended) make the title explicit about the Ns
    # Example: "N_total=199; N_by_cond: CR=70, ..."
    n_by_cond_str = ", ".join([f"{k}={v}" for k, v in n_by_cond.items()])
    title = f"{title} | N_total={n_total} ({n_by_cond_str})"



    # Choose topomap times
    times = [0.2, 0.3, 0.6]

    # Joint plot (butterfly + topo) like MNE report
    fig = evoked.plot_joint(
        times=times,
        title=title,
        show=False,
        ts_args=dict(spatial_colors=True),
    )
    return fig


# ----------------------------
# ICA topo extraction (PNG preferred)
# ----------------------------

#def _load_excluded_ica_topos_from_png(
#    bids_root: Path,
#    subject: str,
#    session: str,
#    excluded_components: Sequence[int],
#) -> Dict[int, str]:
#    """
#    Load ICA excluded component topographies that were saved as PNG by nice_preprocessing.

#    Expected:
#      {bids_root}/derivatives/nice_preprocessing/ica_topos/sub-XXX/ses-M0/eeg/sub-XXX_ses-M0_ica_comp-000.png
#    Returns:
#      comp -> dataURI
#    """
#    out: Dict[int, str] = {}
#
#    topo_dir = (
#        bids_root
#        / "derivatives"
#        / "nice_preprocessing"
#        / "ica_topos"
#        / f"sub-{subject}"
#        / f"ses-{session}"
#        / "eeg"
#    )
#    if not topo_dir.exists():
#        return out
#
#    for comp in excluded_components:
#        png = topo_dir / f"sub-{subject}_ses-{session}_ica_comp-{int(comp):03d}.png"
#        if not png.exists():
#            continue
#        try:
#            b64 = base64.b64encode(png.read_bytes()).decode("ascii")
#            out[int(comp)] = f"data:image/png;base64,{b64}"
#        except Exception:
#            continue
#
#    return out

def _load_ica_excluded_bundle_png(bids_root: Path, subject: str, session: str) -> Optional[str]:
    """
    Load the single PNG containing all excluded ICA components for one subject/session.
    Your naming: sub-XXX_ses-M0_ica_comp-000.png (bundle image)
    """
    topo_dir = (
        bids_root
        / "derivatives"
        / "nice_preprocessing"
        / "ica_topos"
        / f"sub-{subject}"
        / f"ses-{session}"
        / "eeg"
    )
    if not topo_dir.exists():
        return None

    # Your exact filename pattern:
    cand = topo_dir / f"sub-{subject}_ses-{session}_ica_comp-000.png"
    if cand.exists():
        b64 = base64.b64encode(cand.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    # fallback: first PNG in folder (just in case)
    pngs = sorted(topo_dir.glob("*.png"))
    if pngs:
        b64 = base64.b64encode(pngs[0].read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    return None



def _extract_ica_component_images_from_individual_html(html_path: Path) -> Dict[int, str]:
    """
    Fallback: try to extract embedded component topo images from the individual HTML report.
    Returns comp -> dataURI
    """
    try:
        txt = html_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}

    img_pat = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=]+)")
    comp_pat = re.compile(r"component\s*([0-9]{1,3})", re.IGNORECASE)

    images: Dict[int, str] = {}
    for m in img_pat.finditer(txt):
        start = m.start()
        window = txt[max(0, start - 600):start]
        cm = None
        for cm_i in comp_pat.finditer(window):
            cm = cm_i
        if cm is None:
            continue
        comp = int(cm.group(1))
        if comp not in images:
            images[comp] = m.group(0)

    return images


# ----------------------------
# HTML assembly
# ----------------------------

def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )


def _render_html_page(
    session: str,
    n_subjects: int,
    ica_topos_blocks: str,
    eog_scores_blocks: str,
    erp_blocks: str,
) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Group EEG QC Report - {session}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 0; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 16px; }}
    .muted {{ color: #666; font-size: 0.95rem; }}
    .tabs {{ display: flex; gap: 8px; border-bottom: 1px solid #ddd; margin-bottom: 12px; flex-wrap: wrap; }}
    .tabbtn {{ padding: 10px 12px; border: 1px solid #ddd; border-bottom: none; border-radius: 8px 8px 0 0; background: #f6f6f6; cursor: pointer; }}
    .tabbtn.active {{ background: white; font-weight: 600; }}
    .tab {{ display: none; border: 1px solid #ddd; border-radius: 0 8px 8px 8px; padding: 12px; background: white; }}
    .tab.active {{ display: block; }}
    .subject-card {{ border: 1px solid #eee; border-radius: 10px; padding: 10px; margin: 10px 0; }}
    .subject-title {{ font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }}
    .imgwrap img {{ height: 260px; max-width: 100%; width: auto; display: block; margin: 0; border: 1px solid #eee; border-radius: 8px; background: #fff; object-fit: contain; }}
    .small {{ font-size: 0.9rem; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f1f1f1; margin-left: 8px; font-size: 0.85rem; }}
    .sticky-top {{ position: sticky; top: 0; background: white; z-index: 10; padding-top: 8px; }}
    .toc {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0 0; }}
    .toc a {{ color: #0b66c3; text-decoration: none; }}
    .toc a:hover {{ text-decoration: underline; }}
    .note {{ background: #fff8e6; border: 1px solid #ffe5a3; padding: 10px; border-radius: 10px; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="sticky-top">
      <h1 style="margin: 4px 0;">Group EEG QC Report — Session {session}</h1>
      <div class="muted">Subjects included: {n_subjects}</div>
      <div class="tabs" style="margin-top: 12px;">
        <button class="tabbtn active" data-tab="tab-ica-topos">ICA component topographies excluded</button>
        <button class="tabbtn" data-tab="tab-eog">ICA EOG scores</button>
        <button class="tabbtn" data-tab="tab-erp">Time course EEG (ERP)</button>
      </div>
    </div>

    <div id="tab-ica-topos" class="tab active">
      <div class="note small">
        This section shows <b>only excluded ICA component topographies</b> per subject, in subject order.
        Preferred source: PNG files saved by the preprocessing pipeline.
      </div>
      {ica_topos_blocks}
    </div>

    <div id="tab-eog" class="tab">
      <div class="note small">
        This section reproduces your EOG-score barplot (max |score| across EOG channels per component),
        using values stored in each subject JSON report.
      </div>
      {eog_scores_blocks}
    </div>

    <div id="tab-erp" class="tab">
      <div class="note small">
        This section computes one ERP per subject by averaging the three conditions equally:
        <code>{_html_escape(", ".join(DEFAULT_CONDITIONS))}</code>.
        The plot shows a <b>global</b> time course across all EEG channels (GFP or mean depending on --global-metric).
      </div>
      {erp_blocks}
    </div>

  </div>

  <script>
    const btns = document.querySelectorAll('.tabbtn');
    const tabs = document.querySelectorAll('.tab');
    btns.forEach(b => {{
      b.addEventListener('click', () => {{
        btns.forEach(x => x.classList.remove('active'));
        tabs.forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        const t = document.getElementById(b.dataset.tab);
        if (t) t.classList.add('active');
      }});
    }});
  </script>
</body>
</html>
"""


def _make_subject_anchor(subject: str) -> str:
    return f"sub-{subject}"


def _render_subject_toc(subjects: Sequence[str]) -> str:
    links = " ".join([f'<a href="#{_make_subject_anchor(s)}">sub-{s}</a>' for s in subjects])
    return f'<div class="toc">{links}</div>' if links else ""


# ----------------------------
# Core scanning
# ----------------------------

def _collect_subject_sessions(
    reports_root: Path,
    epochs_root: Path,
    sessions: Optional[Sequence[str]] = None,
) -> Dict[str, List[SubjectSessionPaths]]:
    if sessions is None:
        sessions = DEFAULT_SESSIONS

    out: Dict[str, List[SubjectSessionPaths]] = {s: [] for s in sessions}

    for sub_dir in sorted(reports_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        subject = sub_dir.name.replace("sub-", "")

        for sess in sessions:
            sess_dir = sub_dir / f"ses-{sess}" / "eeg"
            if not sess_dir.exists():
                continue

            json_candidates = sorted(sess_dir.glob("*_report.json")) + sorted(sess_dir.glob("*report.json"))
            if not json_candidates:
                continue
            json_path = json_candidates[0]

            html_candidates = sorted(sess_dir.glob("*_report.html")) + sorted(sess_dir.glob("*report.html"))
            html_path = html_candidates[0] if html_candidates else None

            epochs_dir = epochs_root / f"sub-{subject}" / f"ses-{sess}" / "eeg"
            epo = _find_epochs_fif(epochs_dir)

            out[sess].append(
                SubjectSessionPaths(
                    subject=subject,
                    session=sess,
                    json_path=json_path,
                    html_path=html_path,
                    epochs_fif=epo,
                )
            )

    for sess in list(out.keys()):
        out[sess].sort(key=lambda x: _safe_int_subject(x.subject))
        if len(out[sess]) == 0:
            out.pop(sess, None)

    return out


# ----------------------------
# Build session report
# ----------------------------

def _build_session_report(
    session: str,
    entries: List[SubjectSessionPaths],
    out_dir: Path,
    epochs_conditions: Sequence[str],
    bids_root: Path,
    global_metric: str = "gfp",
) -> Path:
    subjects_order = [e.subject for e in entries]
    toc_html = _render_subject_toc(subjects_order)

    ica_blocks = [toc_html]
    eog_blocks = [toc_html]
    erp_blocks = [toc_html]

    for e in entries:
        try:
            rep = _load_json(e.json_path)
        except Exception:
            rep = {}

        excluded, eog_excl, eog_scores, _eog_ch = _extract_ica_excluded_and_eog(rep)


        # ---------- ICA Topos (single bundle PNG per subject) ----------
        bundle_uri = _load_ica_excluded_bundle_png(
            bids_root=bids_root,
            subject=e.subject,
            session=e.session,
        )

        if bundle_uri is not None:
            ica_html = f"""
            <div class="imgwrap">
            <img class="ica-bundle" src="{bundle_uri}" alt="ICA excluded components sub-{e.subject}"/>
            </div>
            """
        else:
            ica_html = '<div class="small muted">ICA excluded topo PNG not found.</div>'

        ica_blocks.append(
            f"""
            <div class="subject-card" id="{_make_subject_anchor(e.subject)}">
            <div class="subject-title">sub-{e.subject}<span class="pill">ses-{session}</span></div>
            {ica_html}
            </div>
            """
        )


        # ---------- ICA EOG Scores ----------
        if eog_scores is not None:
            try:
                fig = _plot_eog_scores_barplot(eog_scores, eog_excl, title_suffix=f"(sub-{e.subject})")
                img = _b64_from_matplotlib_fig(fig)
                eog_blocks.append(
                    f"""
                    <div class="subject-card" id="eog-{_make_subject_anchor(e.subject)}">
                      <div class="subject-title">sub-{e.subject}<span class="pill">ses-{session}</span></div>
                      <div class="small muted">EOG excluded: {eog_excl}</div>
                      <div class="imgwrap"><img src="{img}" alt="ICA EOG scores sub-{e.subject}"/></div>
                    </div>
                    """
                )
            except Exception:
                eog_blocks.append(
                    f"""
                    <div class="subject-card" id="eog-{_make_subject_anchor(e.subject)}">
                      <div class="subject-title">sub-{e.subject}<span class="pill">ses-{session}</span></div>
                      <div class="small muted">Could not generate EOG score plot (malformed scores).</div>
                    </div>
                    """
                )
        else:
            eog_blocks.append(
                f"""
                <div class="subject-card" id="eog-{_make_subject_anchor(e.subject)}">
                  <div class="subject-title">sub-{e.subject}<span class="pill">ses-{session}</span></div>
                  <div class="small muted">No EOG scores found in JSON.</div>
                </div>
                """
            )

        # ---------- ERP ----------
        if e.epochs_fif is not None and e.epochs_fif.exists():
            title = f"ERP (equal average of {len(epochs_conditions)} conditions) — sub-{e.subject} ses-{session}"
            fig = _compute_subject_erp_plot(
                e.epochs_fif,
                epochs_conditions,
                title=title,
                n_topomaps=3,
            )

            if fig is not None:
                img = _b64_from_matplotlib_fig(fig)
                erp_blocks.append(
                    f"""
                    <div class="subject-card" id="erp-{_make_subject_anchor(e.subject)}">
                      <div class="subject-title">sub-{e.subject}<span class="pill">ses-{session}</span></div>
                      <div class="small muted">Epochs: {_html_escape(e.epochs_fif.name)}</div>
                      <div class="imgwrap"><img src="{img}" alt="ERP sub-{e.subject}"/></div>
                    </div>
                    """
                )
            else:
                erp_blocks.append(
                    f"""
                    <div class="subject-card" id="erp-{_make_subject_anchor(e.subject)}">
                      <div class="subject-title">sub-{e.subject}<span class="pill">ses-{session}</span></div>
                      <div class="small muted">Could not compute ERP (missing conditions or read error).</div>
                    </div>
                    """
                )
        else:
            erp_blocks.append(
                f"""
                <div class="subject-card" id="erp-{_make_subject_anchor(e.subject)}">
                  <div class="subject-title">sub-{e.subject}<span class="pill">ses-{session}</span></div>
                  <div class="small muted">No epochs file found for this session.</div>
                </div>
                """
            )

    html = _render_html_page(
        session=session,
        n_subjects=len(entries),
        ica_topos_blocks="\n".join(ica_blocks),
        eog_scores_blocks="\n".join(eog_blocks),
        erp_blocks="\n".join(erp_blocks),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"group_qc_report_ses-{session}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate group QC HTML reports per session.")
    parser.add_argument(
        "--reports-root",
        type=Path,
        required=True,
        help="Root directory containing individual reports (JSON/HTML), e.g. .../derivatives/nice_preprocessing/reports",
    )
    parser.add_argument(
        "--epochs-root",
        type=Path,
        required=True,
        help="Root directory containing preprocessed epochs, e.g. .../derivatives/nice_preprocessing/epochs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory where group HTML reports will be written.",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        default=",".join(DEFAULT_SESSIONS),
        help="Comma-separated list of sessions to build (default: M0,M12,M24,M36,M48,M60).",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=",".join(DEFAULT_CONDITIONS),
        help="Comma-separated list of condition names to average for ERP.",
    )
    parser.add_argument(
        "--global-metric",
        type=str,
        default="gfp",
        choices=["gfp", "mean"],
        help="Global ERP metric across all EEG channels: 'gfp' (std across channels) or 'mean' (mean across channels).",
    )

    args = parser.parse_args()

    reports_root: Path = args.reports_root
    epochs_root: Path = args.epochs_root
    out_dir: Path = args.out_dir

    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    if not reports_root.exists():
        raise SystemExit(f"reports_root does not exist: {reports_root}")
    if not epochs_root.exists():
        raise SystemExit(f"epochs_root does not exist: {epochs_root}")

    # reports_root = .../02_BIDS/derivatives/nice_preprocessing/reports
    # -> bids_root = .../02_BIDS
    try:
        bids_root = reports_root.parents[2]
    except Exception:
        raise SystemExit("Could not infer bids_root from reports_root. Please check your path structure.")

    by_session = _collect_subject_sessions(
        reports_root=reports_root,
        epochs_root=epochs_root,
        sessions=sessions,
    )

    if not by_session:
        raise SystemExit("No sessions found with JSON reports. Check your reports_root path and naming.")

    out_dir.mkdir(parents=True, exist_ok=True)

    built = []
    for sess, entries in by_session.items():
        out_path = _build_session_report(
            session=sess,
            entries=entries,
            out_dir=out_dir,
            epochs_conditions=conditions,
            bids_root=bids_root,
            global_metric=args.global_metric,
        )
        built.append(out_path)

    print("Generated reports:")
    for p in built:
        print(f"  - {p}")


if __name__ == "__main__":
    main()

