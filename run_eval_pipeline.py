"""Evaluation pipeline: prepares eval (test-clean) data only."""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# Force evaluation mode
os.environ["MODE"] = "eval"

MODULE_SEQUENCE = [
    "FYP_BRPE.synthesize_dataset",
    "FYP_BRPE.extract_logmel",
    "FYP_BRPE.make_eval_csv",
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
    print("\nEval pipeline complete.")

if __name__ == "__main__":
    main()