
#B+

eeg-preprocess \
--bids-root /network/iss/cenir/analyse/meeg/LIBERATE/02_BIDS \
--config ./configs/config_minimal.yaml \
--subjects 003 023 052 077 101 110 159 168 172 191 211 226 254 296 354 \
--sessions M0 M12 M24 M36 M48 M60 \
--tasks FCSRT

#controls

eeg-preprocess \
--bids-root /network/iss/cenir/analyse/meeg/LIBERATE/02_BIDS \
--config ./configs/config_minimal.yaml \
--subjects 011 029 058 081 107 118 150 163 177 215 235 250 267 298 369 \
--sessions M0 M12 M24 M36 M48 M60 \
--tasks FCSRT


#progressors

eeg-preprocess \
--bids-root /network/iss/cenir/analyse/meeg/LIBERATE/02_BIDS \
--config ./configs/config_minimal.yaml \
--subjects 050 086 094 102 113 120 175 183 216 240 256 284 300 326 355 \
--sessions M0 M12 M24 M36 M48 M60 \
--tasks FCSRT






#identify flat channels in one epochs file

conda activate mne
python - << 'EOF'
import mne
import numpy as np

epochs = mne.read_epochs(
    "/network/iss/cenir/analyse/meeg/LIBERATE/02_BIDS/derivatives/"
    "nice_preprocessing/epochs/sub-334/ses-M0/eeg/"
    "sub-334_ses-M0_task-FCSRT_proc-clean_desc-cleaned_epo.fif",
    preload=True,
)

# Écart-type par canal (plus c'est petit, plus le canal est "plat")
stds = epochs.get_data().std(axis=(0, 2))

pairs = sorted(zip(epochs.ch_names, stds), key=lambda x: x[1])
print("Canaux les plus plats :")
for ch, s in pairs[:10]:
    print(f"{ch:10s}  std = {s:.4e}")
EOF


#open one raw file

from mne_bids import read_raw_bids

raw = read_raw_bids(bids_root='network/iss/cenir/analyse/meeg/LIBERATE/02_BIDS', subject='001', session='M0', task='FCSRT', run='03')

# save it 

raw.save('/network/iss/home/kenza.bennis/Documents/insight/nice-preprocessing/sub-01_ses-01_task-rest_run-01_raw.fif', overwrite=True)

/network/iss/home/kenza.bennis/Documents/insight/nice-preprocessing/sub-01_ses-01_task-rest_run-01_raw.fif