import os
from pathlib import Path
import pandas as pd
from common_config import COMMON, get_paths

# script to prepare eval (test-clean) metadata csv for eval pipeline
def main():
    # load paths and set up output directory
    root = Path(__file__).parents[1]
    paths = get_paths(os.getenv("MODE", "eval"))
    meta = root / paths["REVERB_DIR"] / "synthesis_metadata.csv"
    out_dir = root / paths["SPLIT_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)

	# read metadata and write out test.csv
    df = pd.read_csv(meta)
    out_csv = out_dir / "test.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")

if __name__ == "__main__":
    main()