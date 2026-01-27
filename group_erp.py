from pathlib import Path
import mne

BASE_EPOCHS = Path(
    "/network/iss/cenir/analyse/meeg/LIBERATE/02_BIDS/"
    "derivatives/nice_preprocessing/epochs"
)

OUT_DIR = Path(
    "/network/iss/cenir/analyse/meeg/LIBERATE/02_BIDS/"
    "derivatives/nice_preprocessing/group_erp"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = ["ses-M0", "ses-M12", "ses-M24", "ses-M36", "ses-M48", "ses-M60"]
TASK = "FCSRT"

CONDS = [
    "Stimulus/CatNewRepeated/CR",
    "Stimulus/CatNewUnique/CR",
    "Stimulus/CatOld/Hit",
]

def list_subjects(base: Path) -> list[str]:
    return sorted([p.name for p in base.glob("sub-*") if p.is_dir()])

def find_epochs_file(sub: str, ses: str) -> Path:
    return (
        BASE_EPOCHS / sub / ses / "eeg"
        / f"{sub}_{ses}_task-{TASK}_proc-clean_desc-cleaned_epo.fif"
    )

def main():
    subjects = list_subjects(BASE_EPOCHS)
    if len(subjects) == 0:
        raise RuntimeError(f"No subjects found under {BASE_EPOCHS}")

    print(f"Found {len(subjects)} subjects in {BASE_EPOCHS}")

    for ses in SESSIONS:
        evokeds_by_cond = {c: [] for c in CONDS}

        # keep track for sanity / reporting
        included_by_cond = {c: [] for c in CONDS}

        for sub in subjects:
            epo_fif = find_epochs_file(sub, ses)
            if not epo_fif.exists():
                continue

            epochs = mne.read_epochs(epo_fif, preload=False)

            # For each condition, include the subject only if that condition exists and has >=1 epoch
            for c in CONDS:
                if c in epochs.event_id and len(epochs[c]) > 0:
                    ev = epochs[c].average()
                    ev = ev.copy().pick("eeg", exclude="bads")  # drop bads; no interpolation
                    evokeds_by_cond[c].append(ev)
                    included_by_cond[c].append(sub)

        print(f"\n=== {ses} ===")
        for c in CONDS:
            print(f"  {c}: n_subjects={len(evokeds_by_cond[c])}")

        # Compute grand averages per condition (intersection of channels across subjects)
        grand_avgs = []
        for c in CONDS:
            evs = evokeds_by_cond[c]
            if len(evs) == 0:
                print(f"[WARN] {ses}: no data for '{c}' -> skipped")
                continue

            # Keep only channels common to all subjects contributing to THIS condition
            mne.channels.equalize_channels(evs)

            gav = mne.grand_average(evs)
            gav.comment = c  # keep condition name in the output file
            print(f"    -> common channels for '{c}': {len(gav.ch_names)}")
            grand_avgs.append(gav)

        if len(grand_avgs) == 0:
            print(f"[WARN] {ses}: nothing to save")
            continue

        out_fif = OUT_DIR / f"grand_average_{ses}_task-{TASK}_3conds_intersection-ave.fif"
        mne.write_evokeds(out_fif, grand_avgs, overwrite=True)
        print(f"[SAVED] {out_fif}")

if __name__ == "__main__":
    main()
