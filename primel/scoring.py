from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import protein_dict
from .fsfp_dataset import MutantSequenceData
from .mutations import apply_mutations, parse_mutations
from .trainer import RankingTrainer


def protein_from_labels(name: str, wild_type: str, labels: list[str]) -> dict:
    rows = []
    for label in labels:
        muts = parse_mutations(label, wild_type)
        canonical = ":".join(mut.label() for mut in muts)
        rows.append(
            {
                "mutant": canonical,
                "wt_aas": "".join(mut.wt for mut in muts),
                "mt_aas": "".join(mut.mt for mut in muts),
                "positions": tuple(mut.pos for mut in muts),
                "mutated_sequence": apply_mutations(wild_type, muts),
                "DMS_score": 0.0,
                "DMS_score_bin": 0,
            }
        )
    df = pd.DataFrame(rows).set_index("mutant", drop=False)
    return protein_dict(name, wild_type, df)


@torch.no_grad()
def score_protein(model, tokenizer, protein: dict, batch_size: int, device, mask: bool = False) -> pd.Series:
    data = MutantSequenceData(protein, tokenizer, mask=mask, device=device)
    loader = DataLoader(data, batch_size=batch_size, collate_fn=data.collate)
    trainer = RankingTrainer(model, log_metrics=[])
    model.eval()
    scores = []
    for batch in tqdm(loader, desc="Scoring variants"):
        scores.append(trainer.predict(batch).detach().cpu())
    scores = torch.cat(scores).numpy()
    return pd.Series(scores, index=protein["df"].index, name="prediction")


def score_labels(model, tokenizer, wild_type: str, labels: list[str], batch_size: int, device, mask: bool = False) -> pd.DataFrame:
    protein = protein_from_labels("candidate_variants", wild_type, labels)
    scores = score_protein(model, tokenizer, protein, batch_size=batch_size, device=device, mask=mask)
    df = protein["df"].copy()
    df["prediction"] = scores
    df = df.sort_values("prediction", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def save_prediction_table(df: pd.DataFrame, path: str | Path, top_k: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if top_k:
        top_path = path.with_name(path.stem + f"_top{top_k}" + path.suffix)
        df.head(top_k).to_csv(top_path, index=False)
