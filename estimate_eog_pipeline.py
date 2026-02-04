from pathlib import Path
import mne
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATASET_ROOT = Path("/data/datasets/ds003483")  # <-- adjust if needed
OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# File discovery (FIF + CTF, exclude derivatives)
# ---------------------------------------------------------------------

fif_files = [
    p
    for p in DATASET_ROOT.rglob("*_meg.fif")
    if p.is_file() and "derivatives" not in str(p) and ".git" not in str(p)
]

ctf_files = [
    p
    for p in DATASET_ROOT.rglob("*.ds")
    if p.is_dir() and "derivatives" not in str(p) and ".git" not in str(p)
]

data_paths = sorted(fif_files + ctf_files)

print(f"Found {len(data_paths)} FIF/CTF files")

# ---------------------------------------------------------------------
# Loader (aligned with reference style)
# ---------------------------------------------------------------------

def process_file(data_path: Path):
    subject_id = data_path.stem
    print(f"\nProcessing {subject_id}")

    try:
        if data_path.suffix == ".fif":
            raw = mne.io.read_raw_fif(
                data_path,
                preload=True,
                verbose=False,
                on_split_missing="ignore"
            )
        elif data_path.suffix == ".ds":
            raw = mne.io.read_raw_ctf(
                data_path,
                preload=True,
                verbose=False
            )
        else:
            print(f"  Unsupported file format: {data_path}")
            return None
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return None

    raw.pick_types(meg=True)
    sfreq = float(raw.info["sfreq"])

    return raw, sfreq, subject_id

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

for data_path in data_paths:

    result = process_file(data_path)
    if result is None:
        continue

    raw, sfreq, subject_id = result

    # Simple sanity-check plot: Global Field Power
    data = raw.get_data()
    gfp = np.std(data, axis=0)

    plt.figure(figsize=(10, 3))
    plt.plot(gfp)
    plt.title(f"{subject_id} – Global Field Power")
    plt.xlabel("Samples")
    plt.ylabel("STD (a.u.)")
    plt.tight_layout()

    out_path = OUT_DIR / f"{subject_id}_gfp.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"  Saved: {out_path}")
