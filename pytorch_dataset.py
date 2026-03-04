"""PyTorch Dataset/DataLoader for LSTM training.

Returns variable length full-file sequences  for LSTM training. a tuple (seq, label) where:
  - seq: Tensor shape (T, F) where T = n_frames (varies per file), F = n_mels
  - label: scalar RT60

TODO: complete this, test and refine
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from common_config import COMMON, get_paths
import pandas as pd

class PyTorchDataset(Dataset):
    """"""
    
    def __init__(self, split_csv, mode="train"):
        self.mode = mode
        self.df = pd.read_csv(split_csv)
        self.segment_frames = int(COMMON.get("SEGMENT_FRAMES", 0))

        root = Path(__file__).parents[1]
        if mode == "test":
            resolved_mode = "eval"
        elif mode in ("val", "train"):
            resolved_mode = "train"
        else:
            resolved_mode = mode

        paths = get_paths(os.getenv("MODE", resolved_mode))

        self.feature_dir = (root / paths["FEATURE_DIR"]).resolve()
        self.reverb_dir = (root / paths["REVERB_DIR"]).resolve()

        # Force train stats for eval/test to avoid leakage
        if resolved_mode == "eval":
            self.stats_json = (root / get_paths("train")["STATS_JSON"]).resolve()
        else:
            self.stats_json = (root / paths["STATS_JSON"]).resolve()

        if not self.stats_json.exists():
            raise FileNotFoundError(f"Stats file not found: {self.stats_json}. Run compute_stats.py first.")
        stats = json.loads(self.stats_json.read_text())
        self.mean = np.array(stats["mean"], dtype=np.float32)[:, None]   # (n_mels, 1)
        self.std = np.array(stats["std"], dtype=np.float32)[:, None]
        self.std[self.std == 0.0] = 1.0  # avoid division by zero

    def _map_out_to_feature(self, out_path):
        p = Path(out_path).resolve()
        try:
            rel = p.relative_to(self.reverb_dir)
        except ValueError:
            raise FileNotFoundError(
                f"Output path {p} is not under REVERB_DIR {self.reverb_dir}."
            )
        return (self.feature_dir / rel).with_suffix(".npy")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_path = self._map_out_to_feature(row["out"])
        
        # load feature: shape (n_mels, n_frames)
        feat = np.load(str(feat_path))
        if feat.ndim != 2:
            raise RuntimeError(f"Expected 2D array (n_mels, n_frames), got shape {feat.shape} for {feat_path}")

        # segment/crop in time axis (frames)
        # train: random crop; val/eval/test: center crop
        T = feat.shape[1]
        seg = self.segment_frames
        if seg > 0 and T > seg:
            if self.mode == "train":
                start = np.random.randint(0, T - seg + 1)
            else:
                start = (T - seg) // 2
            feat = feat[:, start:start + seg]
        
        # normalize per mel band
        feat = (feat - self.mean) / (self.std + 1e-12)
        
        # transpose to (T, F) for LSTM: (n_frames, n_mels)
        seq = torch.from_numpy(feat.T).float()
        
        # label: RT60 (scalar)
        label = torch.tensor(float(row["rt60"]), dtype=torch.float32)
        
        return seq, label


def pad_collate(batch):
    """Collate function that pads variable-length sequences to max length in batch.
    
    Returns:
        batch_seq: (B, T_max, F) padded sequences
        labels: (B,) RT60 labels
        lengths: (B,) actual lengths before padding (for pack_padded_sequence)
    """
    seqs, labels = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)
    n_mels = seqs[0].shape[1]
    
    # create zero-padded batch tensor
    batch_seq = seqs[0].new_zeros(len(seqs), max_len, n_mels)
    for i, s in enumerate(seqs):
        batch_seq[i, :s.shape[0], :] = s
    
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return batch_seq, labels, lengths


def make_dataloader(split_csv, batch_size=16, shuffle=True, num_workers=2, mode="train"):
    """Create DataLoader with padding collate for variable-length sequences.
    
    Args:
        split_csv: path to train.csv / val.csv / test.csv
        batch_size: number of samples per batch
        shuffle: whether to shuffle (typically True for train, False for val/test)
        num_workers: number of data loading workers
        mode: "train" / "val" / "test" (for reference, not used in current impl)
    
    Returns:
        DataLoader yielding (batch_seq, labels, lengths)
    """
    ds = PyTorchDataset(split_csv=split_csv, mode=mode)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pad_collate,
        pin_memory=True
    )
