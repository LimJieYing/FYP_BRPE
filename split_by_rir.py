import os
from pathlib import Path
import random
import pandas as pd
import json
from FYP_BRPE.common_config import COMMON, get_paths

MODE = os.getenv("MODE", "train")
PATHS = get_paths(MODE)

ROOT = Path(__file__).parents[1]
SYNTH_META = ROOT / PATHS["REVERB_DIR"] / "synthesis_metadata.csv"
OUT_DIR = ROOT / PATHS["SPLIT_DIR"]
RATIOS = COMMON["SPLIT_RATIOS"]
SEED = COMMON["SEED"]

# split the synthesized dataset into train/val by RIR (no leakage of same RIR in both sets)
#random split of RIR ids (filenames) into train/val sets according to specified ratio
def make_splits(rir_ids, ratios, seed=None):
    if seed is not None:
        random.seed(seed)
    ids = list(rir_ids)
    random.shuffle(ids)
    n = len(ids)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train = set(ids[:n_train])
    val = set(ids[n_train : n_train + n_val])
    return {"train": train, "val": val}  #moved eval due to new dataset 

# build the train/val/test splits and write out csv files with metadata for each split
def build_splits(df, splits, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rset in splits.items():
        dsplit = df[df["rir"].apply(lambda r: Path(r).name).isin(rset)].reset_index(drop=True)
        csv_out = out_dir / f"{split_name}.csv"
        dsplit.to_csv(csv_out, index=False)
        print(f"{split_name}: {len(dsplit)} samples -> {csv_out}")
    (out_dir / "rir_splits.json").write_text(json.dumps({k: list(v) for k, v in splits.items()}))


def main():
    if SEED is not None:
        random.seed(SEED)
    if not SYNTH_META.exists():
        raise RuntimeError(f"Metadata not found: {SYNTH_META}")
    df = pd.read_csv(SYNTH_META)
    # extract unique RIR ids (filenames)
    df["rir_id"] = df["rir"].apply(lambda r: Path(r).name)
    rir_groups = list(df.groupby("rir_id").groups.keys())

    splits = make_splits(rir_groups, RATIOS, seed=SEED)
    build_splits(df, splits, OUT_DIR)


if __name__ == "__main__":
    main()