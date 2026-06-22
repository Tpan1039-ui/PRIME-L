from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from learn2learn.algorithms import MAML
from peft import PeftModel
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from .lora_maml import pack_lora_layers, replace_modules
from .losses import listwise_ranking_loss, pairwise_ranking_loss
from .plm import forward_logits
from .utils import ensure_dir


def get_optimizer(name: str, lr: float, params):
    params = list(filter(lambda p: p.requires_grad, params))
    if not params:
        return None
    if name == "sgd":
        return optim.SGD(params, lr=lr)
    if name == "nag":
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr)
    if name == "adadelta":
        return optim.Adadelta(params, lr=lr)
    if name == "adam":
        return optim.Adam(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


class RankingTrainer:
    def __init__(
        self,
        model,
        optimizer: str = "adam",
        lr: float = 1e-4,
        epochs: int = 100,
        max_grad_norm: float = 3.0,
        eval_metric: str = "spearmanr",
        log_metrics: list[str] | None = None,
        save_dir: str | Path | None = None,
        patience: int = 10,
        margin: float = 1.0,
        pair_fn: str = "hinge",
    ):
        self.model = model
        self.optimizer = get_optimizer(optimizer, lr, model.parameters())
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.eval_metric = eval_metric
        self.log_metrics = log_metrics or ["spearmanr", "ndcg", "topk_pr"]
        self.save_dir = Path(save_dir) if save_dir else None
        self.patience = patience
        self.margin = margin
        self.pair_fn = pair_fn
        self.curr_epoch = 0
        self.best_epoch = 0
        self.best_score = float("-inf")
        self.logs = defaultdict(list)

    def save_states(self):
        if self.save_dir is None:
            return
        ensure_dir(self.save_dir)
        self.model.save_pretrained(self.save_dir)
        torch.save(dict(self.logs), self.save_dir / "logs.pkl")

    def predict(self, batch):
        logits = forward_logits(self.model, batch["sequences"])
        log_probs = torch.log_softmax(logits, dim=-1)
        predicts = []
        for inv_idx, positions, wt_aas, mt_aas in zip(
            batch["inv_seq_idx"], batch["positions"], batch["wt_aas"], batch["mt_aas"]
        ):
            log_prob = log_probs[inv_idx]
            predict = log_prob[positions, mt_aas] - log_prob[positions, wt_aas]
            predicts.append(predict.sum().unsqueeze(0))
        return torch.cat(predicts)

    def compute_loss(self, batch):
        predicts = self.predict(batch)
        predicts = predicts[batch["inv_list_idx"]]
        targets = batch["targets"][batch["inv_list_idx"]]
        list_size = batch["inv_list_idx"].shape[1]
        if list_size == 1:
            return F.mse_loss(predicts, targets)
        if list_size == 2:
            return pairwise_ranking_loss(predicts[:, 0], predicts[:, 1], targets[:, 0], targets[:, 1], self.pair_fn, self.margin)
        return listwise_ranking_loss(predicts, targets)

    def train_step(self, batch):
        if self.optimizer is None:
            raise RuntimeError("No trainable parameters are available for training.")
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def compute_metrics(self, predicts, targets, labels):
        logs = {}
        predicts_np = predicts.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        labels = labels.detach().cpu()
        for metric in self.log_metrics:
            if metric == "spearmanr":
                value = spearmanr(predicts_np, targets_np).statistic
                logs[metric] = 0.0 if value != value else float(value)
            elif metric == "ndcg":
                std_tgts = minmax_scale(targets_np.reshape(1, -1), (0, 5), axis=1)
                logs[metric] = float(ndcg_score(std_tgts, predicts_np.reshape(1, -1)))
            elif metric == "topk_pr":
                k = min(len(predicts), 30)
                indices = predicts.detach().cpu().topk(k).indices
                logs[metric] = float(torch.count_nonzero(labels[indices]).item() / k)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return logs

    def evaluate_epoch(self, eval_iter):
        self.model.eval()
        predicts, targets, labels = [], [], []
        with torch.no_grad():
            for batch in tqdm(eval_iter, desc="Evaluating"):
                predicts.append(self.predict(batch).cpu())
                targets.append(batch["targets"].cpu())
                labels.append(batch["labels"].cpu())
        predicts = torch.cat(predicts)
        targets = torch.cat(targets)
        labels = torch.cat(labels)
        return predicts, self.compute_metrics(predicts, targets, labels)

    def train_epoch(self, train_iter):
        self.model.train()
        total = 0.0
        for batch in tqdm(train_iter, desc=f"Training epoch {self.curr_epoch + 1}"):
            total += self.train_step(batch)
        loss = total / max(1, len(train_iter))
        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer is not None else 0.0
        return {"train_loss": loss, "lr": lr}

    def __call__(self, train_iter, eval_iter=None):
        for _ in range(self.epochs):
            logs = self.train_epoch(train_iter)
            for key, value in logs.items():
                self.logs[key].append(value)
            self.curr_epoch += 1
            if eval_iter is None:
                continue
            _, eval_logs = self.evaluate_epoch(eval_iter)
            for key, value in eval_logs.items():
                self.logs[key].append(value)
            score = eval_logs[self.eval_metric]
            if score > self.best_score:
                self.best_epoch = self.curr_epoch
                self.best_score = score
                self.save_states()
            elif self.curr_epoch - self.best_epoch >= self.patience:
                break
        if eval_iter is None:
            self.best_epoch = self.curr_epoch
            self.save_states()
        return dict(self.logs)


class MetaRankingTrainer(RankingTrainer):
    def __init__(self, model, adapt_lr: float = 5e-3, first_order: bool = True, **kwargs):
        super().__init__(model, **kwargs)
        if isinstance(model, PeftModel):
            self.adapter_name, adapter = pack_lora_layers(model)
            self.adapter = MAML(adapter, adapt_lr, first_order=first_order)
        else:
            self.model = MAML(model, adapt_lr, first_order=first_order, allow_nograd=True)

    def fast_adapt(self, adapt_batch, eval_batch, training=True):
        if isinstance(self.model, PeftModel):
            cloned_adapter = self.adapter.clone()
            replace_modules(self.model, self.adapter_name, cloned_adapter.module)
            adapt = cloned_adapter.adapt
        else:
            backup = self.model
            self.model = self.model.clone()
            adapt = self.model.adapt
        for batch in adapt_batch:
            adapt(self.compute_loss(batch))
        if training:
            output = self.compute_loss(eval_batch)
        else:
            with torch.no_grad():
                self.model.eval()
                output = self.predict(eval_batch)
                self.model.train()
        if isinstance(self.model, PeftModel):
            replace_modules(self.model, self.adapter_name, self.adapter.module)
        else:
            self.model = backup
        return output

    def train_step(self, batch):
        losses = []
        self.optimizer.zero_grad()
        for adapt_batch, eval_batch in zip(batch["adapt_batches"], batch["eval_batches"]):
            loss = self.fast_adapt(adapt_batch, eval_batch)
            if loss.isfinite():
                loss.backward()
                losses.append(float(loss.item()))
        if not losses:
            return 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(1.0 / len(losses))
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return sum(losses) / len(losses)
