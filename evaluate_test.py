import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from pathlib import Path
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from common_config import COMMON, get_paths
from pytorch_dataset import make_dataloader
#from lstm_model_basic import LSTMEstimator
from lstm_attention_model import LSTMEstimator


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROOT = Path(__file__).parents[1]
    paths = get_paths(os.getenv("MODE", "eval"))
    test_csv = str(ROOT / paths["SPLIT_DIR"] / "test.csv")
    batch_size = 16
    num_workers = 4

    test_loader = make_dataloader(test_csv, batch_size=batch_size, shuffle=False, num_workers=num_workers, mode="eval")

    # input feature dim dynamically
    sample_seq, _ = test_loader.dataset[0]
    n_mels = sample_seq.shape[1]

    model = LSTMEstimator(
        n_mels=n_mels,
        hidden_size=512,
        num_layers=3,
        bidirectional=True,
        dropout=0.2
    ).to(device)

    ckpt_path = ROOT / "best_lstm_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError("best_lstm_model.pth not found. Train the model first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_seq, labels, lengths in test_loader:
            batch_seq = batch_seq.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            preds = model(batch_seq, lengths).squeeze(-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            mse = F.mse_loss(preds, labels, reduction="sum")
            mae = F.l1_loss(preds, labels, reduction="sum")

            total_mse += mse.item()
            total_mae += mae.item()
            n_samples += labels.size(0)

    mse = total_mse / n_samples
    mae = total_mae / n_samples
    rmse = (mse ** 0.5)

    print(f"Test MSE:  {mse:.6f}")
    print(f"Test MAE:  {mae:.6f}")
    print(f"Test RMSE: {rmse:.6f}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    errors = all_preds - all_labels

    # current visualisations, scatter and histo, TODO: will add more?? 
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1,3 , figsize=(12, 5))

    # lhs: Scatterplot, predicted vs true
    axes[0].scatter(all_labels, all_preds, alpha=0.5, s=10)
    axes[0].plot([all_labels.min(), all_labels.max()],
                     [all_labels.min(), all_labels.max()],
                     "r--", lw=2, label="Ideal")
    axes[0].set_xlabel("True RT60")
    axes[0].set_ylabel("Predicted RT60")
    axes[0].set_title("Predicted vs True RT60")
    axes[0].legend()

    # mid, histogram of errors
    axes[1].hist(errors, bins=50, edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Prediction Error")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Histogram of Errors")
    
	# rhs, violin plot 
    bin_edges = np.linspace(all_labels.min(), all_labels.max(), 6)  # 5 bins
    bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
    bin_idx = np.digitize(all_labels, bin_edges[1:-1], right=False)
    plot_df = pd.DataFrame({
        "RT60 Bin": [bin_labels[i] for i in bin_idx],
        "Error": errors
    })

    sns.violinplot(
        data=plot_df,
        x="RT60 Bin",
        y="Error",
        inner="quartile",
        cut=0,
        ax=axes[2]
    )
    axes[2].axhline(0, color="r", linestyle="--", lw=2)
    axes[2].set_title("Error Distribution by RT60 Bin")
    axes[2].tick_params(axis="x", rotation=30)
    

    plt.tight_layout()
    plt.savefig("test_results.png")
    plt.show()
    print("Saved test_results.png, test results visualization")
    
if __name__ == "__main__":
    main()