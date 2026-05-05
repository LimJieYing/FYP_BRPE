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

	# Load test data
    ROOT = Path(__file__).parents[1]
    paths = get_paths("eval")
    test_csv = str(ROOT / paths["SPLIT_DIR"] / "test.csv")
    batch_size = 16
    num_workers = 4

    test_loader = make_dataloader(test_csv, batch_size=batch_size, shuffle=False, num_workers=num_workers, mode="eval")

    # input feature dim dynamically
    sample_seq, _ = test_loader.dataset[0]
    n_mels = sample_seq.shape[1]

	# model setup
    model = LSTMEstimator(
        n_mels=n_mels,
        hidden_size=256,
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

	# eval loop
    # we compute MSE and MAE for the entire test set
	#  store all predictions and labels for further analysis and visualization.
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
    pearson_r = np.corrcoef(all_labels, all_preds)[0, 1]
    print(f"Pearson correlation (r): {pearson_r:.4f}")

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
    
    # Identify and print outliers
    abs_errors = np.abs(errors)
    sorted_indices = np.argsort(abs_errors)[::-1]  # descending order
    
    
    print("top 10 outliers (largest prediction errors):")
   
    for rank, idx in enumerate(sorted_indices[:10], start=1):
        if idx < len(test_loader.dataset.df):
            row = test_loader.dataset.df.iloc[idx]
            speech_path = row.get("speech", "N/A")
            reverb_out_path = row.get("out", "N/A")
            rir_path = row.get("rir", "N/A")
            true_rt60 = all_labels[idx]
            pred_rt60 = all_preds[idx]
            error = errors[idx]
            print(f"\n{rank}. Speech: {speech_path}")
            print(f"   RIR: {rir_path}")
            print(f"   Reverb Output: {reverb_out_path}")
            print(f"   True RT60: {true_rt60:.4f}, Pred RT60: {pred_rt60:.4f}, Error: {error:.4f}")
    print("="*80 + "\n")
    
    print("top 10 best performing (smallest prediction errors):")
    
    for rank, idx in enumerate(sorted_indices[-10:][::-1], start=1):
        if idx < len(test_loader.dataset.df):
            row = test_loader.dataset.df.iloc[idx]
            speech_path = row.get("speech", "N/A")
            reverb_out_path = row.get("out", "N/A")
            rir_path = row.get("rir", "N/A")
            true_rt60 = all_labels[idx]
            pred_rt60 = all_preds[idx]
            error = errors[idx]
            print(f"\n{rank}. Speech: {speech_path}")
            print(f"   RIR: {rir_path}")
            print(f"   Reverb Output: {reverb_out_path}")
            print(f"   True RT60: {true_rt60:.4f}, Pred RT60: {pred_rt60:.4f}, Error: {error:.4f}")

    print("\n" + "="*80)
    print("OUTLIER PATTERN ANALYSIS:")
    print("="*80)
    
    outlier_indices = sorted_indices[:10]
    best_indices = sorted_indices[-10:]
    
    outlier_rt60s = all_labels[outlier_indices]
    best_rt60s = all_labels[best_indices]
    
    print(f"\nOutlier RT60 range: {outlier_rt60s.min():.4f} - {outlier_rt60s.max():.4f}")
    print(f"Best performer RT60 range: {best_rt60s.min():.4f} - {best_rt60s.max():.4f}")
    print(f"Mean outlier RT60: {outlier_rt60s.mean():.4f}")
    print(f"Mean best RT60: {best_rt60s.mean():.4f}")
    
    # Check if room dimensions are available
    print("\nOutlier room dimensions:")
    for idx in outlier_indices[:3]:
        if idx < len(test_loader.dataset.df):
            row = test_loader.dataset.df.iloc[idx]
            room_dim = row.get("room_dim", "N/A")
            print(f"  {room_dim}")
    
    print("\nBest performer room dimensions:")
    for idx in best_indices[:3]:
        if idx < len(test_loader.dataset.df):
            row = test_loader.dataset.df.iloc[idx]
            room_dim = row.get("room_dim", "N/A")
            print(f"  {room_dim}")
    
    # Check absorption coefficients
    print("\nOutlier absorption coefficients:")
    for idx in outlier_indices[:3]:
        if idx < len(test_loader.dataset.df):
            row = test_loader.dataset.df.iloc[idx]
            absorption = row.get("absorption_uniform", "N/A")
            print(f"  {absorption}")
    
    print("\nBest performer absorption coefficients:")
    for idx in best_indices[:3]:
        if idx < len(test_loader.dataset.df):
            row = test_loader.dataset.df.iloc[idx]
            absorption = row.get("absorption_uniform", "N/A")
            print(f"  {absorption}")
    
    # Summary stats
    outlier_absorptions = []
    best_absorptions = []
    for idx in outlier_indices:
        if idx < len(test_loader.dataset.df):
            row = test_loader.dataset.df.iloc[idx]
            abs_val = row.get("absorption_uniform")
            if abs_val is not None and abs_val != "N/A":
                outlier_absorptions.append(abs_val)
    
    for idx in best_indices:
        if idx < len(test_loader.dataset.df):
            row = test_loader.dataset.df.iloc[idx]
            abs_val = row.get("absorption_uniform")
            if abs_val is not None and abs_val != "N/A":
                best_absorptions.append(abs_val)
    
    if outlier_absorptions:
        print(f"\nMean outlier absorption: {np.mean(outlier_absorptions):.6f}")
    if best_absorptions:
        print(f"Mean best performer absorption: {np.mean(best_absorptions):.6f}")

    print("="*80 + "\n")

if __name__ == "__main__":
    main()