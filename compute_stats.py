# Compute per-mel mean/std across training .npy feature files, save stats.json.
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from common_config import COMMON, get_paths

from FYP_BRPE.common_config import COMMON

MODE = os.getenv("MODE", "train")
PATHS = get_paths(MODE)

ROOT = Path(__file__).parents[1]
FEATURE_ROOT = (ROOT / PATHS["FEATURE_DIR"]).resolve()
REVERB_ROOT = (ROOT / PATHS["REVERB_DIR"]).resolve()
SPLIT_CSV = (ROOT / PATHS["SPLIT_DIR"] / "train.csv").resolve()
OUT_PATH = (ROOT / PATHS["STATS_JSON"]).resolve()
BATCH = 1000


def map_outs_to_feature_paths(df):
    files = []
    for out_wav in df["out"].tolist():
        p = Path(out_wav).resolve()
        # require that out paths live under REVERB_ROOT
        try:
            rel = p.relative_to(REVERB_ROOT)
        except Exception:
            raise FileNotFoundError(f"Reverberant path {p} is not under REVERB_ROOT {REVERB_ROOT}")
        feat_path = (FEATURE_ROOT / rel).with_suffix(".npy")
        if not feat_path.exists():
            raise FileNotFoundError(f"Expected feature missing: {feat_path}")
        files.append(str(feat_path))
    if not files:
        raise RuntimeError("No feature files found for the training split.")
    return files


def accumulate_stats(file_list, batch_size=BATCH):
    mean = None
    m2 = None
    count = 0
    for f in tqdm(file_list, desc="Computing stats"):
        data = np.load(f)            # shape (n_mels, n_frames)
        frames = data.T.reshape(-1, data.shape[0])  # (total_frames, n_mels)
        for i in range(0, frames.shape[0], batch_size):
            b = frames[i : i + batch_size]
            if b.size == 0:
                continue
            if mean is None:
                mean = np.mean(b, axis=0)
                m2 = np.var(b, axis=0) * b.shape[0]
                count = b.shape[0]
            else:
                new_count = count + b.shape[0]
                new_mean = (mean * count + b.sum(axis=0)) / new_count
                m2 = m2 + np.var(b, axis=0) * b.shape[0] + (mean - new_mean) ** 2 * count * b.shape[0] / new_count
                mean = new_mean
                count = new_count
    std = np.sqrt(m2 / count)
    return mean, std


def main():
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Train split not found: {SPLIT_CSV}")
    df = pd.read_csv(SPLIT_CSV)
    files = map_outs_to_feature_paths(df)
    mean, std = accumulate_stats(files)
    stats = {"mean": mean.tolist(), "std": std.tolist(), "n_mels": int(len(mean))}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(stats))
    print(f"Wrote stats to {OUT_PATH}")


if __name__ == "__main__":
    main()