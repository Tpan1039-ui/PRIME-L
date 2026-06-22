from __future__ import annotations

import math
import random
from itertools import combinations

import torch
from torch.utils.data import DataLoader, Dataset


class ProteinSequenceData(Dataset):
    def __init__(self, sequences, tokenizer, device=None):
        self.sequences = list(sequences)
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def collate(self, raw_batch):
        batch = self.tokenizer(list(raw_batch), return_tensors="pt", padding=True, return_length=True)
        return batch.to(self.device)


class MutantSequenceData(Dataset):
    def __init__(self, protein, tokenizer, mask: bool = False, device=None):
        self.wild_type = protein["wild_type"]
        df = protein["df"]
        if mask:
            self.sequences = {}
            for positions in set(df["positions"]):
                mutant = list(self.wild_type)
                for position in positions:
                    mutant[position] = "<mask>"
                self.sequences[positions] = "".join(mutant)
        else:
            self.sequences = [self.wild_type]
        self.wt_aas = df["wt_aas"].to_list()
        self.mt_aas = df["mt_aas"].to_list()
        self.positions = df["positions"].to_list()
        self.targets = df["DMS_score"].astype(float).to_list()
        if "DMS_score_bin" in df.columns:
            self.labels = df["DMS_score_bin"].astype(int).to_list()
        else:
            threshold = df["DMS_score"].median()
            self.labels = (df["DMS_score"] > threshold).astype(int).to_list()
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.wt_aas[idx], self.mt_aas[idx], self.positions[idx], self.targets[idx], self.labels[idx]

    def collate(self, raw_batch):
        wt_aas, mt_aas, positions, scores, labels = zip(*raw_batch)
        if isinstance(self.sequences, dict):
            unique_pos = {pos: i for i, pos in enumerate(dict.fromkeys(positions))}
            inv_idx = torch.tensor([unique_pos[pos] for pos in positions], device=self.device)
            sequences = [self.sequences[pos] for pos in unique_pos]
        else:
            inv_idx = torch.zeros(len(positions), dtype=torch.long, device=self.device)
            sequences = self.sequences
        sequences = self.tokenizer(sequences, return_tensors="pt", padding=True).to(self.device)
        tokenized_wt = self.tokenizer(list(wt_aas), add_special_tokens=False)["input_ids"]
        tokenized_mt = self.tokenizer(list(mt_aas), add_special_tokens=False)["input_ids"]
        return {
            "sequences": sequences,
            "inv_seq_idx": inv_idx,
            "wt_aas": tokenized_wt,
            "mt_aas": tokenized_mt,
            "positions": [torch.tensor(pos, device=self.device) + 1 for pos in positions],
            "targets": torch.tensor(scores, dtype=torch.float32, device=self.device),
            "labels": torch.tensor(labels, dtype=torch.long, device=self.device),
        }


class RankingSequenceData(Dataset):
    def __init__(
        self,
        protein,
        tokenizer,
        mask: bool = True,
        list_size: int = 2,
        max_size: int = 10000,
        constructor=MutantSequenceData,
        device=None,
    ):
        self.mutant_data = constructor(protein, tokenizer, mask, device)
        self.list_size = min(list_size, len(self.mutant_data))
        self.max_size = max_size
        self.device = device
        total = math.comb(len(self.mutant_data), self.list_size)
        self.comb_idx = list(combinations(range(len(self.mutant_data)), self.list_size)) if max_size > total else None

    def __len__(self):
        return len(self.comb_idx) if self.comb_idx is not None else self.max_size

    def __getitem__(self, idx):
        if self.comb_idx is not None:
            return self.comb_idx[idx]
        return random.sample(range(len(self.mutant_data)), self.list_size)

    def collate(self, comb_idx):
        comb_idx = torch.tensor(comb_idx, device=self.device)
        unique_mt, inv_idx = torch.unique(comb_idx, return_inverse=True)
        raw_batch = [self.mutant_data[int(i)] for i in unique_mt]
        batch = self.mutant_data.collate(raw_batch)
        batch["inv_list_idx"] = inv_idx
        return batch


class MetaRankingSequenceData(Dataset):
    def __init__(
        self,
        protein_splits,
        tokenizer,
        adapt_batch_size: int,
        eval_batch_size: int,
        adapt_steps: int = 5,
        mask: str = "train",
        list_size: int = 2,
        training: bool = True,
        constructor=MutantSequenceData,
        device=None,
    ):
        self.support_iters = []
        self.query_iters = []
        for support, query in protein_splits:
            support_data = RankingSequenceData(
                support,
                tokenizer,
                mask=mask in {"train", "all"},
                list_size=list_size,
                max_size=adapt_steps * adapt_batch_size,
                constructor=constructor,
                device=device,
            )
            self.support_iters.append(
                DataLoader(support_data, batch_size=adapt_batch_size, shuffle=True, collate_fn=support_data.collate)
            )
            if training:
                query_data = RankingSequenceData(
                    query,
                    tokenizer,
                    mask=mask in {"train", "all"},
                    list_size=list_size,
                    max_size=eval_batch_size,
                    constructor=constructor,
                    device=device,
                )
            else:
                query_data = constructor(query, tokenizer, mask=mask in {"eval", "all"}, device=device)
            self.query_iters.append(DataLoader(query_data, batch_size=eval_batch_size, collate_fn=query_data.collate))

    def __len__(self):
        return len(self.query_iters)

    def __getitem__(self, idx):
        return [batch for batch in self.support_iters[idx]], next(iter(self.query_iters[idx]))

    def collate(self, raw_batch):
        adapt_batches, eval_batches = zip(*raw_batch)
        return {"adapt_batches": adapt_batches, "eval_batches": eval_batches}
