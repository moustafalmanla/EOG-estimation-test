"""
EOG Estimation & Quality Assurance Pipeline for MEG
--------------------------------------------------
This script estimates a proxy EOG signal from raw MEG recordings
when no reliable EOG channel is assumed, following a QA-driven,
uncertainty-aware pipeline.

Outputs:
- Frontal evidence plot
- ICA component plots
- Estimated EOG proxy (if justified)
- Optional comparison with existing EOG channel
- QA summary text files

Author: Moustafa Almanla
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.stats import kurtosis

# ===============================
# USER PARAMETERS
# ===============================

DATA_DIR = Path("data")
OUT_DIR = Path("output")

LOW_FREQ = 0.5
HIGH_FREQ = 15.0
RESAMPLE_FREQ = 200

N_ICA = 30
FRONTAL_Z_THRESHOLD = 2.5
MIN_CONFIDENCE = 0.6

OUT_DIR.mkdir(exist_ok=True)

# ===============================
# HELPER FUNCTIONS
# ===============================

def detect_frontal_evidence(raw):
    picks = mne.pick_types(raw.info, meg='mag')
    frontal = mne.pick_channels_regexp(raw.ch_names, regexp='^MEG0')
    picks = np.intersect1d(picks, frontal)

    data = raw.get_data(picks)
    rms = np.sqrt(np.mean(data ** 2, axis=0))
    z = (rms - rms.mean()) / rms.std()
    return rms, z


def score_ica_component(ic_signal, ic_topo):
    score = 0.0

    score += np.clip(kurtosis(ic_signal) / 10.0, 0, 1)
    score += np.clip(np.std(ic_signal) / np.max(np.abs(ic_signal)), 0, 1)

    frontal_weight = np.mean(np.abs(ic_topo))
    score += np.clip(frontal_weight / np.max(np.abs(ic_topo)), 0, 1)

    return score / 3.0

# ===============================
# MAIN PIPELINE
# ===============================

for fif_file in DATA_DIR.rglob("*_meg.fif"):

    subj = fif_file.parents[1].name
    out_subj = OUT_DIR / subj
    out_subj.mkdir(exist_ok=True)

    print(f"Processing {subj}")

    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
    raw.pick_types(meg=True)

    # -------- Data hygiene --------
    raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)
    raw.notch_filter(50, verbose=False)
    raw.resample(RESAMPLE_FREQ)

    # -------- Frontal evidence --------
    rms, z = detect_frontal_evidence(raw)

    plt.figure(figsize=(10, 4))
    plt.plot(z, lw=0.8)
    plt.axhline(FRONTAL_Z_THRESHOLD, color='r', linestyle='--')
    plt.title("Frontal RMS Z-score (Ocular Evidence Test)")
    plt.tight_layout()
    plt.savefig(out_subj / "frontal_evidence.png")
    plt.close()

    if np.max(z) < FRONTAL_Z_THRESHOLD:
        with open(out_subj / "qa_summary.txt", "w") as f:
            f.write("NO RELIABLE OCULAR EVIDENCE DETECTED\n")
        continue

    # -------- ICA hypothesis generation --------
    ica = mne.preprocessing.ICA(
        n_components=N_ICA,
        random_state=97,
        max_iter="auto"
    )
    ica.fit(raw)

    sources = ica.get_sources(raw).get_data()
    mixing = ica.get_components()

    scores = [score_ica_component(sources[i], mixing[:, i])
              for i in range(sources.shape[0])]

    best_ic = int(np.argmax(scores))
    confidence = scores[best_ic]

    ica.plot_components(show=False)
    plt.savefig(out_subj / "ica_components.png")
    plt.close("all")

    with open(out_subj / "qa_summary.txt", "w") as f:
        f.write(f"Best IC: {best_ic}\n")
        f.write(f"Confidence score: {confidence:.2f}\n")

        if confidence < MIN_CONFIDENCE:
            f.write("EOG ESTIMATION AMBIGUOUS\n")
            continue

        f.write("EOG ESTIMATION ACCEPTED\n")

    # -------- EOG proxy --------
    eog_proxy = sources[best_ic]

    plt.figure(figsize=(10, 4))
    plt.plot(eog_proxy, lw=0.8)
    plt.title("Estimated EOG Proxy")
    plt.tight_layout()
    plt.savefig(out_subj / "eog_proxy.png")
    plt.close()

    # -------- Optional comparison with EOG channel --------
    eog_picks = mne.pick_types(raw.info, eog=True)

    if len(eog_picks) > 0:
        eog_ref = raw.get_data(eog_picks[0])[0]

        eog_ref_z = (eog_ref - eog_ref.mean()) / eog_ref.std()
        eog_proxy_z = (eog_proxy - eog_proxy.mean()) / eog_proxy.std()

        plt.figure(figsize=(10, 3))
        plt.plot(eog_ref_z)
        plt.title("Reference EOG Channel (z-scored)")
        plt.tight_layout()
        plt.savefig(out_subj / "eog_reference.png")
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(eog_ref_z, label="Reference EOG", alpha=0.7)
        plt.plot(eog_proxy_z, label="Estimated EOG Proxy", alpha=0.7)
        plt.legend()
        plt.title("EOG Proxy vs Reference")
        plt.tight_layout()
        plt.savefig(out_subj / "eog_proxy_vs_reference.png")
        plt.close()

        corr = np.corrcoef(eog_ref_z, eog_proxy_z)[0, 1]
        with open(out_subj / "eog_correlation.txt", "w") as f:
            f.write(f"Correlation (proxy vs reference): {corr:.3f}\n")
