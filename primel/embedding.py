from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .plm import forward_last_hidden


class SequenceDataset(Dataset):
    def __init__(self, labels: list[str], sequences: list[str], tokenizer, device=None):
        self.labels = labels
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.labels[idx], self.sequences[idx]

    def collate(self, raw_batch):
        labels, sequences = zip(*raw_batch)
        batch = self.tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            return_special_tokens_mask=True,
        )
        return list(labels), batch.to(self.device)


def mean_pool(hidden: torch.Tensor, token_batch) -> torch.Tensor:
    attention = token_batch["attention_mask"].bool()
    if "special_tokens_mask" in token_batch:
        residue_mask = attention & ~token_batch["special_tokens_mask"].bool()
    else:
        residue_mask = attention.clone()
        residue_mask[:, 0] = False
        residue_mask[torch.arange(residue_mask.size(0)), attention.sum(1) - 1] = False
    denom = residue_mask.sum(1).clamp_min(1).unsqueeze(1)
    return (hidden * residue_mask.unsqueeze(-1)).sum(1) / denom


@torch.no_grad()
def embed_sequences(model, tokenizer, labels: list[str], sequences: list[str], batch_size: int, device) -> tuple[list[str], torch.Tensor]:
    dataset = SequenceDataset(labels, sequences, tokenizer, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate)
    all_labels: list[str] = []
    vectors = []
    model.eval()
    for batch_labels, token_batch in tqdm(loader, desc="Embedding sequences"):
        model_inputs = {k: v for k, v in token_batch.items() if k != "special_tokens_mask"}
        hidden = forward_last_hidden(model, model_inputs)
        vectors.append(mean_pool(hidden, token_batch).detach().cpu())
        all_labels.extend(batch_labels)
    return all_labels, torch.cat(vectors, dim=0)


def records_from_protein(protein: dict) -> tuple[list[str], list[str], torch.Tensor]:
    df = protein["df"]
    labels = df.index.to_list()
    sequences = df["mutated_sequence"].to_list()
    scores = torch.tensor(df["DMS_score"].astype(float).to_numpy(), dtype=torch.float32)
    return labels, sequences, scores


def records_from_labels(wild_type: str, labels: list[str]) -> tuple[list[str], list[str]]:
    from .mutations import apply_mutations, parse_mutations

    canonical = []
    sequences = []
    for label in labels:
        muts = parse_mutations(label, wild_type)
        canonical.append(":".join(mut.label() for mut in muts))
        sequences.append(apply_mutations(wild_type, muts))
    return canonical, sequences


def select_by_sequence_similarity(model, tokenizer, query_sequence: str, tasks: list[dict], k: int, batch_size: int, device):
    labels = ["__query__"] + [task["name"] for task in tasks]
    sequences = [query_sequence] + [task["wild_type"] for task in tasks]
    _, vectors = embed_sequences(model, tokenizer, labels, sequences, batch_size, device)
    query = vectors[:1]
    corpus = vectors[1:]
    scores = torch.cosine_similarity(query.expand_as(corpus), corpus, dim=1)
    top = scores.topk(min(k, len(tasks))).indices.tolist()
    return [(tasks[i], float(scores[i])) for i in top]


def save_embeddings(labels: list[str], vectors: torch.Tensor, path) -> None:
    pd.DataFrame(vectors.numpy(), index=labels).to_pickle(path)
