import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from common_config import COMMON, get_paths
import os

from pytorch_dataset import make_dataloader
#from lstm_model_basic import LSTMEstimator
from lstm_attention_model import LSTMEstimator


# training loop for one epoch, returns average loss over the epoch
def train_one_epoch(model, loader, optimizer, device, task="regression", grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x, y, lengths = batch
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x, lengths).squeeze(-1)
        # not used, for experiments 
        #weights = 1.0 + 2.0 * (y > 1.0).float() # test, double weight for rt60 > 1s
        loss = F.mse_loss(preds, y)
        
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

# evaluation loop, returns average loss over the dataset
@torch.no_grad()
def evaluate(model, loader, device, task="regression"):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        x, y, lengths = batch
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        preds = model(x, lengths).squeeze(-1)
        #weights = 1.0 + 2.0 * (y > 1.0).float()
        loss = F.mse_loss(preds, y)
        
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROOT = Path(__file__).parents[1]
    paths = get_paths(os.getenv("MODE", "train"))
    train_csv = str(ROOT / paths["SPLIT_DIR"] / "train.csv")
    val_csv = str(ROOT / paths["SPLIT_DIR"] / "val.csv")

    batch_size = 32
    num_workers = 4
    epochs = 20
    lr = 5e-4
    weight_decay = 1.5e-4
    grad_clip = 1.0

    train_loader = make_dataloader(train_csv, batch_size=batch_size, shuffle=True, num_workers=num_workers, mode="train")
    val_loader = make_dataloader(val_csv, batch_size=batch_size, shuffle=False, num_workers=num_workers, mode="val")

    # infer input feature dim dynamically
    sample_seq, _ = train_loader.dataset[0]
    n_mels = sample_seq.shape[1]

# same model as lstm
    model = LSTMEstimator(
        n_mels=n_mels,
        hidden_size=256,
        num_layers=3,
        bidirectional=True,
        dropout=0.2
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# scheduler that reduces LR by factor of 0.5 if val loss doesn't improve for 3 epochs, with a minimum LR of 1e-6
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    best_val = float("inf")
    train_history = [] #store train loss for epochs 
    val_history = [] #store val loss for epochs

#training loop, save best model based on val loss, and plot train/val loss curves at the end
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, task="regression", grad_clip=grad_clip)
        train_history.append(train_loss)
        val_loss = evaluate(model, val_loader, device, task="regression")
        val_history.append(val_loss)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_lstm_model.pth")
            print(f"Saved best model (val_loss={best_val:.4f})")
        
    # plot graph for training 
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    sns.lineplot(x=range(1, epochs + 1), y=train_history, label="Train Loss")
    sns.lineplot(x=range(1, epochs + 1), y=val_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Over Time")
    plt.savefig("training_val_loss_plot.png")
    plt.tight_layout()
    plt.show()
    print("Plot saved as 'training_val_loss_plot.png'.")

if __name__ == "__main__":
    main()