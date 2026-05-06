# Compute log-mel spectrograms for reverberant WAVs and save .npy files 
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import os
from common_config import COMMON, get_paths

MODE = os.getenv("MODE", "train")
PATHS = get_paths(MODE)

ROOT = Path(__file__).parents[1]
REVERB_DIR = (ROOT / PATHS["REVERB_DIR"]).resolve()
OUT_DIR = (ROOT / PATHS["FEATURE_DIR"]).resolve()
SR = int(COMMON["SR"])
N_FFT = int(COMMON["N_FFT"])
HOP = int(COMMON["HOP"])
N_MELS = int(COMMON["N_MELS"])
LOG_OFFSET = float(COMMON["LOG_OFFSET"])

#log-mel extraction function
#params taken from common_config, we can use librosa's melspectrogram function
def compute_log_mel(y, sr, n_fft, hop, n_mels, log_offset):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window="hann",
        center=True,
        power=2.0,
        n_mels=n_mels,
    )
    log_mel = np.log(S + log_offset)
    return log_mel.astype(np.float32)

def main():
    files = sorted(list(REVERB_DIR.rglob("*_reverb.wav")))
    if len(files) == 0:
        raise RuntimeError(f"No *_reverb.wav files found under {REVERB_DIR}")
    for p in tqdm(files, desc="Extracting features"):
        #load wav, convert to mono if needed, resample if needed, 
        y, fs = sf.read(str(p))
        if y.ndim > 1:
            y = y.mean(axis=1)
        if fs != SR:
            y = librosa.resample(y.astype(np.float32), fs, SR)
        #compute log-mel and save as .npy
        log_mel = compute_log_mel(y, SR, N_FFT, HOP, N_MELS, LOG_OFFSET)
        rel = p.resolve().relative_to(REVERB_DIR).with_suffix("")
        out_path = OUT_DIR / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # explicitly save shape (n_mels, n_frames) dtype float32
        np.save(str(out_path) + ".npy", log_mel.astype(np.float32))
    print(f"Saved features to {OUT_DIR}")

if __name__ == "__main__":
    main()