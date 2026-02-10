"""PyTorch Dataset/DataLoader for LSTM training.

Returns variable-length sequences suitable for LSTM:
  - seq: Tensor shape (T, F) where T = n_frames (varies per file), F = n_mels
  - label: scalar RT60

Uses pad_collate to batch variable-length sequences.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from FYP_BRPE.common_config import COMMON

ROOT = Path(__file__).parents[1]
FEATURE_DIR = (ROOT / COMMON["FEATURE_DIR"]).resolve()
REVERB_DIR = (ROOT / COMMON["REVERB_DIR"]).resolve()
STATS_JSON = (ROOT / COMMON["STATS_JSON"]).resolve()


class PyTorchDataset(Dataset):
    """Returns full-file sequences (variable length) for LSTM training."""
    
    def __init__(self, split_csv, mode="train"):
        import pandas as pd
        self.df = pd.read_csv(split_csv)
        self.mode = mode
        
        # load normalization stats
        if not STATS_JSON.exists():
            raise FileNotFoundError(f"Stats file not found: {STATS_JSON}. Run compute_stats.py first.")
        stats = json.loads(STATS_JSON.read_text())
        self.mean = np.array(stats["mean"], dtype=np.float32)[:, None]   # (n_mels, 1)
        self.std = np.array(stats["std"], dtype=np.float32)[:, None]
        self.std[self.std == 0.0] = 1.0  # avoid division by zero

    def _map_out_to_feature(self, out_wav_path):
        """Map reverberant WAV path to corresponding .npy feature file."""
        p = Path(out_wav_path).resolve()
        try:
            rel = p.relative_to(REVERB_DIR)
        except ValueError:
            raise FileNotFoundError(
                f"Output path {p} is not under REVERB_DIR {REVERB_DIR}. "
                "Ensure synthesize_dataset writes absolute paths."
            )
        feat_path = (FEATURE_DIR / rel).with_suffix(".npy")
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_path}")
        return feat_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_path = self._map_out_to_feature(row["out"])
        
        # load feature: shape (n_mels, n_frames)
        feat = np.load(str(feat_path))
        if feat.ndim != 2:
            raise RuntimeError(f"Expected 2D array (n_mels, n_frames), got shape {feat.shape} for {feat_path}")
        
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


# Quick test when run directly
if __name__ == "__main__":
    train_csv = ROOT / COMMON["SPLIT_DIR"] / "train.csv"
    if not train_csv.exists():
        print(f"Train CSV not found: {train_csv}")
        print("Run the pipeline first: python scripts/run_pipeline.py")
    else:
        print("Testing DataLoader...")
        dl = make_dataloader(str(train_csv), batch_size=4, num_workers=0, mode="train")
        batch_seq, labels, lengths = next(iter(dl))
        print(f"batch_seq shape: {batch_seq.shape}")  # (B, T_max, F)
        print(f"labels shape: {labels.shape}")        # (B,)
        print(f"lengths: {lengths}")                  # (B,)
        print(f"Sample 0 actual length: {lengths[0]}, padded length: {batch_seq.shape[1]}")
        print("✓ Dataset is ready for LSTM training")