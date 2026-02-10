# Minimal centralized project config. Edit only this file.
from pathlib import Path

ROOT = Path(__file__).parents[1]

COMMON = {
    # paths (relative to project root)
    "SPEECH_DIR": "LibriSpeech/test-clean",
    "RIR_DIR": "data/rir",
    "REVERB_DIR": "data/reverb",
    "FEATURE_DIR": "data/features",
    "SPLIT_DIR": "data/splits",
    "STATS_JSON": "data/features/stats.json",

    # RIR generation
    "NUM_RIRS": 200,
    "FS": 16000,
    "MIN_RT60": 0.2,
    "MAX_RT60": 1.2,
    "MIN_DIM": [3.0, 3.0, 2.8],
    "MAX_DIM": [8.0, 6.0, 3.5],
    "MAX_ORDER": 12,
    "SEED": 1234,

    # synthesis
    "ADD_NOISE": True,
    "SNR_MIN": 10.0,
    "SNR_MAX": 20.0,
    "MAX_FILES": 1000  ,

    # features
    "SR": 16000,
    "N_FFT": 1024,
    "HOP": 256,
    "N_MELS": 80,
    "LOG_OFFSET": 1e-6,

    # splits
    "SPLIT_RATIOS": [0.7, 0.15, 0.15],

    # dataset defaults
    "SEGMENT_FRAMES": 300,
}