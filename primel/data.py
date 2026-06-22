from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .mutations import CANONICAL_AAS, Mutation, apply_mutations, format_mutations, is_wt_label, parse_mutations
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


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python")


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_to_original = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _score_column_from_table(df: pd.DataFrame) -> str:
    col = _first_existing(
        df.columns,
        ("gemme_score", "GEMME_score", "score", "DMS_score", "fitness", "prediction", "pseudo_fitness"),
    )
    if col is not None:
        return col
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        raise ValueError("Could not identify a GEMME score column")
    return numeric_cols[-1]


def _gemme_long_records(df: pd.DataFrame, wild_type: str) -> pd.DataFrame | None:
    mutant_col = _first_existing(df.columns, ("mutant", "mutation", "mutations", "variant"))
    if mutant_col is None:
        return None
    score_col = _score_column_from_table(df)
    rows = []
    for _, row in df.iterrows():
        muts = parse_mutations(row[mutant_col], wild_type)
        score = float(row[score_col])
        rows.append(
            {
                "mutant": format_mutations(muts),
                "raw_mutant": str(row[mutant_col]),
                "coerced_from": "",
                "wt_aas": "".join(mut.wt for mut in muts),
                "mt_aas": "".join(mut.mt for mut in muts),
                "positions": tuple(mut.pos for mut in muts),
                "mutated_sequence": apply_mutations(wild_type, muts),
                "DMS_score": score,
            }
        )
    return pd.DataFrame(rows)


def _position_from_column(col) -> int | None:
    text = str(col)
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return int(digits) - 1


def _gemme_matrix_records(df: pd.DataFrame, wild_type: str) -> pd.DataFrame:
    first_col = df.columns[0]
    rows = []
    for _, row in df.iterrows():
        mt = str(row[first_col]).strip().upper()
        if len(mt) != 1 or mt not in CANONICAL_AAS:
            continue
        for col in df.columns[1:]:
            pos = _position_from_column(col)
            if pos is None or pos < 0 or pos >= len(wild_type):
                continue
            score = row[col]
            if pd.isna(score):
                continue
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue
            wt = wild_type[pos]
            if mt == wt:
                continue
            mut = Mutation(wt, pos, mt)
            rows.append(
                {
                    "mutant": mut.label(),
                    "raw_mutant": mut.label(),
                    "coerced_from": "",
                    "wt_aas": mut.wt,
                    "mt_aas": mut.mt,
                    "positions": (mut.pos,),
                    "mutated_sequence": apply_mutations(wild_type, (mut,)),
                    "DMS_score": score,
                }
            )
    if not rows:
        raise ValueError("Could not parse GEMME data as long table or amino-acid matrix")
    return pd.DataFrame(rows)


def load_gemme_task(path: str | Path, wild_type: str, max_records: int | None = None) -> dict:
    df = read_table(path)
    records = _gemme_long_records(df, wild_type)
    if records is None:
        records = _gemme_matrix_records(df, wild_type)
    records = records.replace([np.inf, -np.inf], np.nan).dropna(subset=["DMS_score"])
    records = records.drop_duplicates(subset=["mutant"], keep="last").set_index("mutant", drop=False)
    if max_records is not None and max_records > 0:
        records = records.iloc[:max_records].copy()
    protein = protein_dict("LPE1439_GEMME", wild_type, records)
    return add_binary_labels(protein)


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
