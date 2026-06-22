from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from .mutations import Mutation, apply_mutations, format_mutations, is_wt_label, parse_mutations
from .utils import atomic_torch_save, ensure_dir


AA_CHARS = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def _clean_sequence(text: str) -> str | None:
    if ">LPE1439" not in text and "MSNKSNDELK" not in text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith(">")]
    seq = "".join(lines).split("*")[0]
    seq = "".join(ch for ch in seq if ch in AA_CHARS)
    return seq or None


def extract_lpe_sequence(workbook: str | Path) -> str:
    workbook = Path(workbook)
    strings: list[str] = []
    raw = pd.read_excel(workbook, sheet_name="Sheet1", header=None)
    strings.extend(str(x) for x in raw.to_numpy().ravel())
    header = pd.read_excel(workbook, sheet_name="Sheet1")
    strings.extend(str(x) for x in header.columns)
    for text in strings:
        seq = _clean_sequence(text)
        if seq:
            return seq
    raise ValueError(f"Could not find LPE1439 sequence in {workbook}")


def _score_column(df: pd.DataFrame) -> str:
    for col in ("Relative_Activity", "Relative activity", "DMS_score", "fitness"):
        if col in df.columns:
            return col
    raise ValueError(f"No activity column found in columns: {list(df.columns)}")


def _mutant_column(df: pd.DataFrame) -> str:
    for col in ("Mutations", "Mutational information", "mutant"):
        if col in df.columns:
            return col
    raise ValueError(f"No mutation column found in columns: {list(df.columns)}")


def table_to_records(df: pd.DataFrame, wild_type: str, include_wt: bool = False) -> pd.DataFrame:
    score_col = _score_column(df)
    mutant_col = _mutant_column(df)
    wt_rows = df[df[mutant_col].map(is_wt_label)]
    wt_score = float(wt_rows.iloc[0][score_col]) if len(wt_rows) else 1.0

    rows = []
    for _, row in df.iterrows():
        raw_label = row[mutant_col]
        if is_wt_label(raw_label):
            if not include_wt:
                continue
            muts = tuple()
        else:
            parsed = parse_mutations(raw_label)
            coerced = []
            mismatches = []
            for mut in parsed:
                actual_wt = wild_type[mut.pos]
                if actual_wt != mut.wt:
                    mismatches.append(f"{mut.label()}=>{actual_wt}{mut.one_based}{mut.mt}")
                coerced.append(Mutation(actual_wt, mut.pos, mut.mt))
            muts = tuple(coerced)
        label = format_mutations(muts)
        score = float(row[score_col])
        rows.append(
            {
                "mutant": label,
                "raw_mutant": str(raw_label),
                "coerced_from": ";".join(mismatches) if not is_wt_label(raw_label) else "",
                "wt_aas": "".join(mut.wt for mut in muts),
                "mt_aas": "".join(mut.mt for mut in muts),
                "positions": tuple(mut.pos for mut in muts),
                "mutated_sequence": apply_mutations(wild_type, muts),
                "DMS_score": score,
                "DMS_score_bin": int(score > wt_score),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No mutation records were parsed")
    return out.set_index("mutant", drop=False)


def protein_dict(name: str, wild_type: str, df: pd.DataFrame) -> dict:
    n_sites = sorted({len(pos) for pos in df["positions"]})
    return {"name": name, "wild_type": wild_type, "df": df, "n_sites": n_sites, "offset": 0}


def load_study_datasets(study_dir: str | Path) -> dict:
    study_dir = Path(study_dir)
    paths = {
        "ala_scan": study_dir / "This study_Ala scan.xlsx",
        "round1": study_dir / "This study_PLA_round1.xlsx",
        "round2": study_dir / "This study_PLA_round2.xlsx",
        "round3": study_dir / "This study_PLA_round3.xlsx",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing study workbooks: {missing}")
    wild_type = extract_lpe_sequence(paths["ala_scan"])
    datasets = {"name": "LPE1439", "wild_type": wild_type, "source_dir": str(study_dir)}
    for name, path in paths.items():
        df = pd.read_excel(path, sheet_name="Sheet2")
        records = table_to_records(df, wild_type, include_wt=False)
        datasets[name] = protein_dict(f"LPE1439_{name}", wild_type, records)
    return datasets


def save_study_datasets(study_dir: str | Path, output_pkl: str | Path, summary_dir: str | Path | None = None) -> dict:
    datasets = load_study_datasets(study_dir)
    atomic_torch_save(datasets, output_pkl)
    if summary_dir:
        summary = ensure_dir(summary_dir)
        for key in ("ala_scan", "round1", "round2", "round3"):
            datasets[key]["df"].to_csv(summary / f"{key}.csv", index=False)
    return datasets


def load_lpe_pkl(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def add_binary_labels(protein: dict, threshold: float | None = None) -> dict:
    protein = protein.copy()
    df = protein["df"].copy()
    if "DMS_score_bin" not in df.columns:
        threshold = float(df["DMS_score"].median()) if threshold is None else threshold
        df["DMS_score_bin"] = (df["DMS_score"] > threshold).astype(int)
    protein["df"] = df
    return protein


def load_proteingym_tasks(path: str | Path, min_size: int = 10) -> list[dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    tasks: list[dict] = []
    for value in obj.values():
        for protein in value:
            if len(protein["df"]) >= min_size:
                tasks.append(add_binary_labels(protein))
    return tasks


def subset_protein(protein: dict, max_records: int | None = None) -> dict:
    if max_records is None or len(protein["df"]) <= max_records:
        return protein
    out = protein.copy()
    out["df"] = protein["df"].iloc[:max_records].copy()
    out["n_sites"] = sorted({len(pos) for pos in out["df"]["positions"]})
    return out


def concatenate_rounds(datasets: dict, rounds: Iterable[str], name: str) -> dict:
    frames = [datasets[round_name]["df"] for round_name in rounds]
    df = pd.concat(frames, axis=0)
    df = df.loc[~df.index.duplicated(keep="last")].copy()
    return protein_dict(name, datasets["wild_type"], df)
