# Minimal centralized project config. Edit only this file.
from pathlib import Path

ROOT = Path(__file__).parents[1]

COMMON = {
    # shared params
    "RIR_DIR": "data/rir",
    "NUM_RIRS": 1500,
    "FS": 16000,
    "MIN_RT60": 0.2,
    "MAX_RT60": 1.2,
    "MIN_DIM": [3.0, 3.0, 2.8],
    "MAX_DIM": [8.0, 6.0, 3.5],
    "MAX_ORDER": 12,
    "SEED": 4323,
    "ADD_NOISE": False,
    "SNR_MIN": 10.0,
    "SNR_MAX": 20.0,
    "MAX_FILES": 16000,
    "SR": 16000,
    "N_FFT": 1024,
    "HOP": 256,
    "N_MELS": 80,
    "LOG_OFFSET": 1e-6,
    "SPLIT_RATIOS": [0.85, 0.15, 0.0],
    "SEGMENT_FRAMES": 300,
    "USE_DECAY_RATE": True,   #append decay-rate 
}

DATASETS = {
    "train": {
        "SPEECH_DIR": "LibriSpeech/train-clean-100",
        "REVERB_DIR": "data/reverb",
        "FEATURE_DIR": "data/features",
        "SPLIT_DIR": "data/splits",
        "STATS_JSON": "data/features/stats.json",
    },
    "eval": {
        "SPEECH_DIR": "LibriSpeech/test-clean",
        "REVERB_DIR": "data/reverb_eval",
        "FEATURE_DIR": "data/features_eval",
        "SPLIT_DIR": "data/splits_eval",
        "STATS_JSON": "data/features_eval/stats.json",
    },
}

def get_paths(mode="train"):
    if mode not in DATASETS:
        raise ValueError(f"Unknown mode: {mode}")
    return DATASETS[mode]