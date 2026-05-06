"""Generate synthetic room impulse responses (RIRs) using pyroomacoustics.

"""
from pathlib import Path
import json
import random
import numpy as np
import soundfile as sf
import pyroomacoustics as pra

from FYP_BRPE.common_config import COMMON

ROOT = Path(__file__).parents[1]
OUT_DIR = ROOT / COMMON["RIR_DIR"]
NUM = int(COMMON["NUM_RIRS"])
FS = int(COMMON["FS"])
MIN_RT60 = float(COMMON["MIN_RT60"])
MAX_RT60 = float(COMMON["MAX_RT60"])
MIN_DIM = COMMON["MIN_DIM"]
MAX_DIM = COMMON["MAX_DIM"]
MAX_ORDER = int(COMMON["MAX_ORDER"])
SEED = COMMON["SEED"]

# calculate uniform absorption coefficient for given RT60 and room dimensions using Sabine's formula
def uniform_absorption_for_rt60(rt60, room_dim):
    # Sabine: RT60 = 0.161 * V / (alpha * S) -> alpha = 0.161 * V / (RT60 * S)
    L, W, H = room_dim
    V = L * W * H
    S = 2 * (L * W + L * H + W * H)
    alpha = 0.161 * V / (max(rt60, 0.05) * S)
    return float(np.clip(alpha, 0.001, 0.999))

#generate a single RIR with random room dimensions, source/mic positions, and absorption for target RT60
def generate_rir(room_dim, rt60, fs, max_order):
    
    alpha = uniform_absorption_for_rt60(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order, absorption=alpha)

    src_pos = [
        random.uniform(0.5, room_dim[0] - 0.5),
        random.uniform(0.5, room_dim[1] - 0.5),
        random.uniform(0.8, min(2.0, room_dim[2] - 0.2)),
    ]
    
    mic_pos = [
        random.uniform(0.5, room_dim[0] - 0.5),
        random.uniform(0.5, room_dim[1] - 0.5),
        random.uniform(0.8, min(2.0, room_dim[2] - 0.2)),
    ]

    room.add_source(src_pos)
    mic = np.array(mic_pos).reshape(3, 1)
    room.add_microphone_array(pra.MicrophoneArray(mic, room.fs))
    room.compute_rir()
    rir = np.array(room.rir[0][0], dtype=np.float32)
    return rir, alpha, src_pos, mic_pos


def main():
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "rirs_metadata.jsonl"

    with open(meta_path, "w", encoding="utf-8") as meta_f:
        for i in range(NUM):
            dims = [
                float(random.uniform(MIN_DIM[0], MAX_DIM[0])),
                float(random.uniform(MIN_DIM[1], MAX_DIM[1])),
                float(random.uniform(MIN_DIM[2], MAX_DIM[2])),
            ]
            # for stratified sampling across RT60 range
			# cycle through NUM_BINS bins and sample uniformly within each bin
            # rt60 = float(random.uniform(MIN_RT60, MAX_RT60))
            NUM_BINS = 10
            bin_size = (MAX_RT60 - MIN_RT60) / NUM_BINS
            bin_idx = i % NUM_BINS  # cycle through bins
            rt60 = float(random.uniform(
				MIN_RT60 + bin_idx * bin_size,
				MIN_RT60 + (bin_idx + 1) * bin_size
			))
            rir, alpha, src_pos, mic_pos = generate_rir(dims, rt60, fs=FS, max_order=MAX_ORDER)

            wav_name = f"rir_{i:05d}.wav"
            wav_path = OUT_DIR / wav_name
            sf.write(str(wav_path), rir, FS)

			# write metadata as JSON lines 
            meta = {
                "wav": wav_path.name,
                "rt60": float(rt60),
                "room_dim": [float(d) for d in dims],
                "absorption_uniform": float(alpha),
                "src_pos": [float(x) for x in src_pos],
                "mic_pos": [float(x) for x in mic_pos],
                "fs": FS,
            }
            meta_f.write(json.dumps(meta) + "\n")

            if (i + 1) % 10 == 0 or (i + 1) == NUM:
                print(f"Generated {i+1}/{NUM} RIRs -> {wav_path.name}")

    print(f"Done. RIRs and metadata saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
