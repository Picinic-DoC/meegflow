# run_qc.py
from math import ceil
from pathlib import Path

from qc_metrics import QCThresholds
from qc_classify import ROIQCConfig
from qc_report import PathsConfig, run_qc_mad

paths = PathsConfig(
    base_epochs=Path("/network/iss/cenir/analyse/meeg/LIBERATE/02_BIDS/derivatives/nice_preprocessing/epochs"),
    out_dir=Path("/network/iss/cenir/analyse/meeg/LIBERATE/02_BIDS/derivatives/nice_preprocessing/qc_automagic_like_mad"),
    task="FCSRT",
    sessions=("ses-M0","ses-M12","ses-M24","ses-M36","ses-M48","ses-M60"),
)

# thresholds used to DEFINE the metrics (Automagic/Pedroni-style starting points)
thr_uv = QCThresholds(oha_uv=50.0, hv_uv=15.0)

ROIS_NUMBERS = {
    "RAS": [13, 14, 5, 4, 6, 224, 223, 215, 214, 7, 207, 206, 205, 198, 197, 196],
    "LAS": [28, 22, 29, 35, 40, 36, 23, 41, 30, 16, 50, 42, 24, 51, 43, 17],
    "RPS": [164, 173, 154, 163, 172, 142, 153, 162, 171, 179, 141, 152, 161, 170, 183, 182, 181, 180, 129],
    "LPS": [66, 72, 78, 77, 76, 88, 87, 86, 85, 84, 99, 98, 97, 96, 59, 65, 71, 75, 100],
    "LPI": [110,109,118,108,117,107,116,125,106,115,124,105,114,123,136,113,122,135,112,121,134,146,111],
    "RPI": [128,127,140,139,151,138,150,160,149,159,169,148,158,168,177,157,167,176,156,166,175,188,199],
    "FP":  [10,11,12,19,20,25,26,27,32,33,34,38,47, 31],
    "PM":  [45,81,132,53,80,90,131,144,60,79,89,130,143,155,52,44,9,186,185,184],
    "RAI": [2,3,220,221,222,211,212,213,202,203,204,194,195,192,193],
    "LAI": [39,47,48,55,61,49,56,62,57,63,58,64,69,70,74],
}

# ROI rule: keep if >= 40% good channels in ROI
ROI_CFGS = {}
for roi_name, roi_nums in ROIS_NUMBERS.items():
    n = len(roi_nums)
    ROI_CFGS[roi_name] = ROIQCConfig(
        max_rbc=0.60,                # <=60% bad
        min_good=ceil(0.40 * n),     # >=40% good
    )

if __name__ == "__main__":
    run_qc_mad(
        paths=paths,
        thresholds_uv=thr_uv,
        rois_numbers=ROIS_NUMBERS,
        roi_cfgs=ROI_CFGS,
        roi_prefix="E",
        mad_k=3.0,                   # median + 3*MAD
        expected_n_channels=256,      # helps document missing channels (221 vs 256)
        mne_log_level="WARNING",
    )

