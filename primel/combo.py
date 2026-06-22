from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .embedding import mean_pool
from .plm import forward_last_hidden
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


class SequenceFitnessDataset(Dataset):
    def __init__(self, labels: list[str], sequences: list[str], targets: torch.Tensor | None, tokenizer, device=None):
        self.labels = labels
        self.sequences = sequences
        self.targets = targets
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        target = float(self.targets[idx]) if self.targets is not None else 0.0
        return self.labels[idx], self.sequences[idx], target

    def collate(self, raw_batch):
        labels, sequences, targets = zip(*raw_batch)
        tokens = self.tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            return_special_tokens_mask=True,
        ).to(self.device)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        return list(labels), tokens, targets


def make_encoder_trainable(model, gradient_checkpointing: bool = True):
    for param in model.parameters():
        param.requires_grad = True
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    return model


def pooled_encoder_output(model, token_batch) -> torch.Tensor:
    model_inputs = {key: value for key, value in token_batch.items() if key != "special_tokens_mask"}
    hidden = forward_last_hidden(model, model_inputs)
    return mean_pool(hidden, token_batch)


def train_full_sequence_model_once(
    model,
    tokenizer,
    train_labels: list[str],
    train_sequences: list[str],
    train_y: torch.Tensor,
    valid_labels: list[str] | None = None,
    valid_sequences: list[str] | None = None,
    valid_y: torch.Tensor | None = None,
    *,
    hidden_dim: int,
    dropout: float,
    lr: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device,
):
    model = make_encoder_trainable(model).to(device)
    head = RegressionHead(model.config.hidden_size, hidden_dim, dropout).to(device)
    y_scaled, mean, std = standardize(train_y)
    train_data = SequenceFitnessDataset(train_labels, train_sequences, y_scaled, tokenizer, device=device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate)
    valid_loader = None
    if valid_labels is not None and valid_sequences is not None and valid_y is not None:
        valid_scaled = (valid_y - mean) / std
        valid_data = SequenceFitnessDataset(valid_labels, valid_sequences, valid_scaled, tokenizer, device=device)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, collate_fn=valid_data.collate)
    params = [param for param in list(model.parameters()) + list(head.parameters()) if param.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    best_loss = float("inf")
    best_epoch = 0
    bad = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        head.train()
        for _, tokens, targets in train_loader:
            optimizer.zero_grad()
            pooled = pooled_encoder_output(model, tokens)
            preds = head(pooled)
            loss = nn.functional.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()
        if valid_loader is None:
            best_epoch = epoch
            continue
        model.eval()
        head.eval()
        losses = []
        with torch.no_grad():
            for _, tokens, targets in valid_loader:
                pooled = pooled_encoder_output(model, tokens)
                preds = head(pooled)
                losses.append(nn.functional.mse_loss(preds, targets).item())
        valid_loss = float(np.mean(losses)) if losses else float("inf")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    metadata = {
        "target_mean": mean,
        "target_std": std,
        "final_epochs": best_epoch or max_epochs,
        "input_dim": int(model.config.hidden_size),
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "encoder_trainable": True,
    }
    return model, head, metadata, best_epoch or max_epochs, best_loss


def fit_full_sequence_model_cv_then_final(
    model_factory,
    tokenizer,
    labels: list[str],
    sequences: list[str],
    targets: torch.Tensor,
    *,
    output_dir: str | Path,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-5,
    max_epochs: int = 200,
    patience: int = 10,
    batch_size: int = 1,
    folds: int = 5,
    seed: int = 666666,
    device=None,
):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(targets)
    fold_epochs = []
    fold_losses = []
    if n >= 3 and folds > 1:
        kfold = KFold(n_splits=min(folds, n), shuffle=True, random_state=seed)
        for train_idx, valid_idx in kfold.split(np.arange(n)):
            model = model_factory()
            _, _, _, best_epoch, best_loss = train_full_sequence_model_once(
                model,
                tokenizer,
                [labels[i] for i in train_idx],
                [sequences[i] for i in train_idx],
                targets[train_idx],
                [labels[i] for i in valid_idx],
                [sequences[i] for i in valid_idx],
                targets[valid_idx],
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
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        final_epochs = max(1, int(round(float(np.mean(fold_epochs)))))
    else:
        final_epochs = max_epochs
    final_model = model_factory()
    final_model, head, metadata, _, _ = train_full_sequence_model_once(
        final_model,
        tokenizer,
        labels,
        sequences,
        targets,
        None,
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
    metadata.update(
        {
            "fold_epochs": fold_epochs,
            "fold_valid_mse": fold_losses,
            "final_epochs": final_epochs,
        }
    )
    output_dir = ensure_dir(output_dir)
    torch.save({"state_dict": head.state_dict(), "metadata": metadata}, output_dir / "regression_head.pt")
    with (output_dir / "regression_head.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    return final_model, head, metadata


@torch.no_grad()
def predict_sequences_with_full_model(model, head, tokenizer, labels: list[str], sequences: list[str], metadata: dict, batch_size: int, device):
    dataset = SequenceFitnessDataset(labels, sequences, None, tokenizer, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate)
    model.eval()
    head.eval()
    predictions = []
    for _, tokens, _ in loader:
        pooled = pooled_encoder_output(model, tokens)
        scaled = head(pooled).cpu()
        predictions.append(scaled)
    scaled = torch.cat(predictions)
    return scaled * metadata["target_std"] + metadata["target_mean"]
