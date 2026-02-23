"""Synthesize reverberant dataset by convolving clean speech (FLAC) with RIRs.

"""
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve
import os
from common_config import COMMON, get_paths

from FYP_BRPE.common_config import COMMON

MODE = os.getenv("MODE", "train")
PATHS = get_paths(MODE)

ROOT = Path(__file__).parents[1]
SPEECH_DIR = ROOT / PATHS["SPEECH_DIR"]
RIR_DIR = ROOT / COMMON["RIR_DIR"]
OUT_DIR = ROOT / PATHS["REVERB_DIR"]
ADD_NOISE = bool(COMMON["ADD_NOISE"])
SNR_MIN = float(COMMON["SNR_MIN"])
SNR_MAX = float(COMMON["SNR_MAX"])
MAX_FILES = int(COMMON["MAX_FILES"])
SEED = COMMON["SEED"]


def load_rirs_metadata(rir_dir: Path):
    meta_path = rir_dir / "rirs_metadata.jsonl"
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta

def build_rir_map(rirs_meta):
    """Map rir filename -> rt60 for labeling."""
    return {Path(m["wav"]).name: m.get("rt60") for m in rirs_meta}

def build_rir_meta_map(rirs_meta):
    """Map rir filename -> full metadata dict."""
    return {Path(m["wav"]).name: m for m in rirs_meta}

def add_noise_at_snr(signal, snr_db):
    rms_signal = np.sqrt(np.mean(signal**2) + 1e-12)
    noise = np.random.randn(signal.shape[0]).astype(np.float32)
    rms_noise = np.sqrt(np.mean(noise**2) + 1e-12)
    target_rms_noise = rms_signal / (10 ** (snr_db / 20))
    scale = target_rms_noise / rms_noise
    return signal + noise * scale

def convolve_and_save(speech_path: Path, rir_path: Path, out_path: Path, snr=None):
    x, fs_x = sf.read(str(speech_path))
    h, fs_h = sf.read(str(rir_path))
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if h.ndim > 1:
        h = np.mean(h, axis=1)

    # resample RIR to speech fs if needed 
    if fs_x != fs_h:
        try:
            import librosa
            h = librosa.resample(h.astype(np.float32), orig_sr=fs_h, target_sr=fs_x)
            fs_h = fs_x
        except Exception as e:
            raise RuntimeError("Sample rate mismatch and librosa not available.") from e

    y = fftconvolve(x.astype(np.float32), h.astype(np.float32), mode="full")
    # Normalize peak to avoid clipping
    peak = float(np.max(np.abs(y)) + 1e-12)
    if peak > 0.999:
        y = y * (0.99 / peak)

    if snr is not None:
        y = add_noise_at_snr(y, snr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y.astype(np.float32), fs_x)

def main():
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
    if not SPEECH_DIR.exists():
        raise RuntimeError(f"Speech directory not found: {SPEECH_DIR}")
    if not RIR_DIR.exists():
        raise RuntimeError(f"RIR directory not found: {RIR_DIR}")
    rirs_meta = load_rirs_metadata(RIR_DIR)
    rir_map = build_rir_map(rirs_meta)
    rir_meta_map = build_rir_meta_map(rirs_meta)
    rir_files = [RIR_DIR / m["wav"] for m in rirs_meta]
    if len(rir_files) == 0:
        raise RuntimeError(f"No RIR wavs found in {RIR_DIR}")
    speech_files = sorted(list(SPEECH_DIR.rglob("*.flac")))
    if MAX_FILES:
        speech_files = speech_files[:MAX_FILES]
    if len(speech_files) == 0:
        raise RuntimeError(f"No FLAC speech files found under {SPEECH_DIR}")
    rows = []
    for i, sp in enumerate(speech_files, start=1):
        rir_choice = random.choice(rir_files)
        snr = None
        if ADD_NOISE and (SNR_MIN is not None and SNR_MAX is not None):
            snr = float(np.random.uniform(SNR_MIN, SNR_MAX))
        try:
            rel = sp.relative_to(SPEECH_DIR)
        except Exception:
            rel = sp.name
        out_base = (OUT_DIR / rel).with_suffix("")
        out_file = out_base.with_name(out_base.name + "_reverb.wav")
        convolve_and_save(sp, rir_choice, out_file, snr=snr)
        rir_name = Path(rir_choice).name
        rt60 = rir_map.get(rir_name)
        meta = rir_meta_map.get(rir_name, {})
        if rt60 is None:
            raise RuntimeError(f"RT60 not found for RIR: {rir_name}")

        rows.append({
            "speech": str(sp.resolve()),
            "rir": str(rir_choice.resolve()),
            "out": str(out_file.resolve()),
            "snr_db": snr,
            "rt60": rt60,
            "room_dim": meta.get("room_dim"),
            "absorption_uniform": meta.get("absorption_uniform"),
            "fs": meta.get("fs")
        })
        if i % 10 == 0 or i == len(speech_files):
            print(f"Processed {i}/{len(speech_files)} -> {out_file}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "synthesis_metadata.csv", index=False)
    print(f"Done. Reverberant files and metadata written to {OUT_DIR}")

if __name__ == "__main__":
    main()
