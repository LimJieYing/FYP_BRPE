import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from pathlib import Path
from common_config import COMMON, get_paths
import os

from pytorch_dataset import make_dataloader
#from lstm_model_basic import LSTMEstimator
from lstm_attention_model import LSTMEstimator

criterion = nn.SmoothL1Loss()

def train_one_epoch(model, loader, optimizer, device, task="regression", grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x, y, lengths = batch
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x, lengths)

        if task == "regression":
            preds = preds.squeeze(-1)
            loss = criterion(preds, y)
        else:
            loss = F.cross_entropy(preds, y)

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, task="regression"):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        x, y, lengths = batch
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        preds = model(x, lengths)

        if task == "regression":
            preds = preds.squeeze(-1)
            loss = criterion(preds, y)
        else:
            loss = F.cross_entropy(preds, y)

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
    epochs = 40
    lr = 5e-4
    weight_decay = 1e-4
    early_patience = 6
    grad_clip = 1.0

    train_loader = make_dataloader(train_csv, batch_size=batch_size, shuffle=True, num_workers=num_workers, mode="train")
    val_loader = make_dataloader(val_csv, batch_size=batch_size, shuffle=False, num_workers=num_workers, mode="val")

    n_mels = int(COMMON["N_MELS"])
    model = LSTMEstimator(n_mels=n_mels, hidden_size=256, num_layers=2, bidirectional=True, dropout=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    best_val = float("inf")
    patience = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, task="regression", grad_clip=grad_clip)
        val_loss = evaluate(model, val_loader, device, task="regression")
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), "best_lstm_model.pth")
            print(f"Saved best model (val_loss={best_val:.4f})")
        else:
            patience += 1
            if patience >= early_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Done.")


if __name__ == "__main__":
    main()