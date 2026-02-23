"""Simple orchestrator: runs all pipeline scripts in sequence.
Edit COMMON_CONFIG here for project-wide paths used by multiple scripts.
"""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# Force training mode
os.environ["MODE"] = "train"

from FYP_BRPE.common_config import COMMON

MODULE_SEQUENCE = [
    "FYP_BRPE.simulate_rirs",
    "FYP_BRPE.synthesize_dataset",
    "FYP_BRPE.extract_logmel",
    "FYP_BRPE.split_by_rir",
    "FYP_BRPE.compute_stats",
]

def call_main(module_name):
    print(f"\n--- RUN: {module_name} ---")
    if module_name in sys.modules:
        mod = importlib.reload(sys.modules[module_name])
    else:
        mod = importlib.import_module(module_name)
    mod.main()
    print(f"--- DONE: {module_name} ---")

def main():
    print(f"Project root: {ROOT.resolve()}")
    for m in MODULE_SEQUENCE:
        call_main(m)
    print("\nPipeline complete.")

if __name__ == "__main__":
    main()