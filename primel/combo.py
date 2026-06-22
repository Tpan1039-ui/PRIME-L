from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import ensure_dir, set_seed


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def standardize(y: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    mean = float(y.mean().item())
    std = float(y.std(unbiased=False).clamp_min(1e-8).item())
    return (y - mean) / std, mean, std


def train_head_once(
    train_x,
    train_y,
    valid_x=None,
    valid_y=None,
    *,
    hidden_dim: int,
    dropout: float,
    lr: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device,
):
    head = RegressionHead(train_x.size(1), hidden_dim, dropout).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    best_state = copy.deepcopy(head.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    bad = 0
    for epoch in range(1, max_epochs + 1):
        head.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(head(x_batch), y_batch)
            loss.backward()
            opt.step()
        if valid_x is None:
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            continue
        head.eval()
        with torch.no_grad():
            valid_loss = nn.functional.mse_loss(head(valid_x.to(device)), valid_y.to(device)).item()
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    return head, best_epoch, best_loss


def fit_cv_then_final(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    *,
    output_dir: str | Path,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-5,
    max_epochs: int = 200,
    patience: int = 10,
    batch_size: int = 4,
    folds: int = 5,
    seed: int = 666666,
    device=None,
):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_scaled, mean, std = standardize(targets)
    n = len(targets)
    fold_epochs = []
    fold_losses = []
    if n >= 3 and folds > 1:
        kfold = KFold(n_splits=min(folds, n), shuffle=True, random_state=seed)
        for train_idx, valid_idx in kfold.split(np.arange(n)):
            _, best_epoch, best_loss = train_head_once(
                embeddings[train_idx],
                y_scaled[train_idx],
                embeddings[valid_idx],
                y_scaled[valid_idx],
                hidden_dim=hidden_dim,
                dropout=dropout,
                lr=lr,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                device=device,
            )
            fold_epochs.append(best_epoch)
            fold_losses.append(best_loss)
        final_epochs = max(1, int(round(float(np.mean(fold_epochs)))))
    else:
        final_epochs = max_epochs
    head, _, _ = train_head_once(
        embeddings,
        y_scaled,
        None,
        None,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        max_epochs=final_epochs,
        patience=patience,
        batch_size=batch_size,
        device=device,
    )
    output_dir = ensure_dir(output_dir)
    metadata = {
        "target_mean": mean,
        "target_std": std,
        "fold_epochs": fold_epochs,
        "fold_valid_mse": fold_losses,
        "final_epochs": final_epochs,
        "input_dim": int(embeddings.size(1)),
        "hidden_dim": hidden_dim,
        "dropout": dropout,
    }
    torch.save({"state_dict": head.state_dict(), "metadata": metadata}, output_dir / "regression_head.pt")
    with (output_dir / "regression_head.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    return head, metadata


@torch.no_grad()
def predict_with_head(head, embeddings: torch.Tensor, metadata: dict, device) -> torch.Tensor:
    head.eval()
    preds = []
    loader = DataLoader(embeddings, batch_size=256)
    for x_batch in loader:
        scaled = head(x_batch.to(device)).cpu()
        preds.append(scaled)
    scaled = torch.cat(preds)
    return scaled * metadata["target_std"] + metadata["target_mean"]


def prediction_table(labels: list[str], predictions: torch.Tensor) -> pd.DataFrame:
    df = pd.DataFrame({"mutant": labels, "prediction": predictions.numpy()})
    df = df.sort_values("prediction", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df
