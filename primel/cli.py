from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .combo import fit_full_sequence_model_cv_then_final, predict_sequences_with_full_model, prediction_table
from .config import load_config, repo_root, resolve_path
from .data import (
    add_binary_labels,
    concatenate_rounds,
    load_lpe_pkl,
    load_gemme_task,
    load_proteingym_tasks,
    save_study_datasets,
    subset_protein,
)
from .embedding import records_from_labels, records_from_protein, select_by_sequence_similarity
from .fsfp_dataset import MetaRankingSequenceData, RankingSequenceData
from .mutations import combine_single_mutations, enumerate_single_mutants, unique_single_mutations
from .plm import apply_lora, count_trainable, load_adapter_model, load_model_and_tokenizer, load_tokenizer
from .scoring import protein_from_labels, save_prediction_table, score_labels
from .trainer import MetaRankingTrainer, RankingTrainer
from .utils import ensure_dir, get_device, set_seed


def cfg_path(cfg: dict, key: str, default: str) -> Path:
    return resolve_path(cfg.get(key, default), repo_root())


def model_args(cfg: dict) -> dict:
    model_cfg = cfg.get("model", {})
    return {
        "model_name": model_cfg.get("name", "AI4Protein/Prime_690M"),
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
        "local_files_only": model_cfg.get("local_files_only", True),
    }


def load_or_prepare(cfg: dict, data_path: Path) -> dict:
    if not data_path.exists():
        save_study_datasets(cfg["study_dir"], data_path, summary_dir=data_path.parent / "summaries")
    return load_lpe_pkl(data_path)


def split_half(protein: dict, seed: int) -> tuple[dict, dict]:
    shuffled = protein["df"].sample(frac=1, random_state=seed)
    mid = max(1, len(shuffled) // 2)
    support = protein.copy()
    query = protein.copy()
    support["df"] = shuffled.iloc[:mid].copy()
    query["df"] = shuffled.iloc[mid:].copy()
    return support, query


def build_pseudo_task(model, tokenizer, wild_type: str, size: int, batch_size: int, device, mask: bool) -> dict:
    labels = enumerate_single_mutants(wild_type)[:size]
    df = score_labels(model, tokenizer, wild_type, labels, batch_size, device, mask=mask)
    protein = protein_from_labels("LPE1439_prime_pseudo", wild_type, df["mutant"].to_list())
    protein["df"]["DMS_score"] = df.set_index("mutant").loc[protein["df"].index, "prediction"].astype(float)
    return add_binary_labels(protein)


def build_meta_tasks(cfg, args, model, tokenizer, lpe_data, device) -> list[dict]:
    protein_gym_tasks: list[dict] = []
    extra_tasks: list[dict] = []
    if args.meta_tasks > 0:
        pg_path = resolve_path(cfg.get("protein_gym_pkl", "../data/merged.pkl"), repo_root())
        proteingym = load_proteingym_tasks(pg_path)
        selected = select_by_sequence_similarity(
            model,
            tokenizer,
            lpe_data["wild_type"],
            proteingym,
            k=args.meta_tasks,
            batch_size=args.embed_batch_size,
            device=device,
        )
        protein_gym_tasks.extend(task for task, _ in selected)
    gemme_path = args.gemme_data or cfg.get("gemme_data")
    if gemme_path:
        extra_tasks.append(load_gemme_task(resolve_path(gemme_path, repo_root()), lpe_data["wild_type"], args.max_gemme_records))
    if args.pseudo_task_size > 0:
        extra_tasks.append(
            build_pseudo_task(
                model,
                tokenizer,
                lpe_data["wild_type"],
                args.pseudo_task_size,
                args.eval_batch_size,
                device,
                args.mask in {"eval", "all"},
            )
        )
    return [subset_protein(task, args.max_aux_records) for task in protein_gym_tasks] + extra_tasks


def cmd_prepare(args):
    cfg = load_config(args.config)
    data_path = resolve_path(args.output or cfg.get("lpe_data_pkl", "data/lpe1439.pkl"), repo_root())
    datasets = save_study_datasets(cfg["study_dir"], data_path, summary_dir=data_path.parent / "summaries")
    print(f"Saved {data_path}")
    print(f"LPE1439 length: {len(datasets['wild_type'])}")
    for key in ("ala_scan", "round1", "round2", "round3"):
        print(f"{key}: {len(datasets[key]['df'])} variants")


def cmd_finetune_ala(args):
    cfg = load_config(args.config)
    data_path = resolve_path(args.data or cfg.get("lpe_data_pkl", "data/lpe1439.pkl"), repo_root())
    out_dir = resolve_path(args.output, repo_root())
    set_seed(args.seed)
    device = get_device(args.force_cpu)
    lpe_data = load_or_prepare(cfg, data_path)
    model, tokenizer = load_model_and_tokenizer(
        **model_args(cfg),
        output_hidden_states=args.meta_tasks > 0 or bool(args.gemme_data or cfg.get("gemme_data")) or args.pseudo_task_size > 0,
    )
    lora_cfg = cfg.get("lora", {})
    model = apply_lora(
        model,
        r=args.lora_r if args.lora_r is not None else lora_cfg.get("r", 16),
        alpha=lora_cfg.get("alpha", None),
        dropout=lora_cfg.get("dropout", 0.1),
        target_modules=lora_cfg.get("target_modules", ["query", "key", "value", "dense"]),
    )
    trainable, total = count_trainable(model)
    print(f"Trainable parameters: {trainable} / {total} ({100 * trainable / total:.3f}%)")
    model.to(device)

    meta_tasks = build_meta_tasks(cfg, args, model, tokenizer, lpe_data, device)
    if meta_tasks:
        print(f"Meta-training on {len(meta_tasks)} auxiliary tasks")
        splits = [split_half(task, args.seed + i) for i, task in enumerate(meta_tasks)]
        meta_data = MetaRankingSequenceData(
            splits,
            tokenizer,
            adapt_batch_size=args.meta_train_batch,
            eval_batch_size=args.meta_eval_batch,
            adapt_steps=args.adapt_steps,
            mask=args.mask,
            list_size=args.list_size,
            training=True,
            device=device,
        )
        meta_loader = DataLoader(meta_data, batch_size=1, shuffle=True, collate_fn=meta_data.collate)
        meta_trainer = MetaRankingTrainer(
            model,
            optimizer=args.optimizer,
            lr=args.learning_rate,
            epochs=args.meta_epochs,
            max_grad_norm=args.max_grad_norm,
            adapt_lr=args.adapt_lr,
            save_dir=out_dir / "meta_adapter",
            patience=args.patience,
        )
        meta_trainer(meta_loader)

    target = lpe_data["ala_scan"]
    train_data = RankingSequenceData(
        target,
        tokenizer,
        mask=args.mask in {"train", "all"},
        list_size=args.list_size,
        max_size=args.max_iter * args.train_batch,
        device=device,
    )
    train_loader = DataLoader(train_data, batch_size=args.train_batch, shuffle=True, collate_fn=train_data.collate)
    trainer = RankingTrainer(
        model,
        optimizer=args.optimizer,
        lr=args.learning_rate,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        save_dir=out_dir,
        patience=args.patience,
    )
    logs = trainer(train_loader)
    tokenizer.save_pretrained(out_dir)
    with (ensure_dir(out_dir) / "primel_training.json").open("w") as handle:
        json.dump(
            {
                "config": cfg,
                "data_path": str(data_path),
                "mask": args.mask,
                "epochs": args.epochs,
                "meta_tasks": [task["name"] for task in meta_tasks],
                "logs": logs,
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"Saved PRIME-L adapter to {out_dir}")


def cmd_predict_single(args):
    cfg = load_config(args.config)
    data_path = resolve_path(args.data or cfg.get("lpe_data_pkl", "data/lpe1439.pkl"), repo_root())
    lpe_data = load_or_prepare(cfg, data_path)
    device = get_device(args.force_cpu)
    model = load_adapter_model(**model_args(cfg), adapter_dir=resolve_path(args.checkpoint, repo_root()))
    tokenizer = load_tokenizer(**model_args(cfg))
    model.to(device)
    labels = enumerate_single_mutants(lpe_data["wild_type"])
    if args.limit:
        labels = labels[: args.limit]
    df = score_labels(model, tokenizer, lpe_data["wild_type"], labels, args.batch_size, device, mask=args.mask in {"eval", "all"})
    out_path = resolve_path(args.output, repo_root())
    save_prediction_table(df, out_path, top_k=args.top_k)
    print(f"Saved {len(df)} single-site predictions to {out_path}")


def default_combo_sizes(round_id: int) -> tuple[int, int]:
    return (2, 3) if round_id == 2 else (3, 5)


def measured_exclude_set(lpe_data: dict, round_id: int, mode: str) -> set[str]:
    if mode == "none":
        return set()
    if mode == "all":
        rounds = ("round1", "round2", "round3")
    else:
        rounds = ("round1",) if round_id == 2 else ("round1", "round2")
    return set().union(*(set(lpe_data[name]["df"].index) for name in rounds))


def candidate_labels_from_args(args, lpe_data: dict, train_protein: dict) -> list[str]:
    if args.candidates:
        raw = pd.read_csv(resolve_path(args.candidates, repo_root()))
        col = "mutant" if "mutant" in raw.columns else raw.columns[0]
        return raw[col].astype(str).to_list()
    if args.single_predictions:
        raw = pd.read_csv(resolve_path(args.single_predictions, repo_root()))
        seed_labels = raw.head(args.seed_top_k)["mutant"].astype(str).to_list()
    else:
        seed_labels = train_protein["df"].index.to_list()
    seeds = unique_single_mutations(seed_labels, lpe_data["wild_type"])
    preset_id = combo_preset_id(args)
    min_sites, max_sites = args.sites if args.sites else default_combo_sizes(preset_id)
    return combine_single_mutations(
        seeds,
        min_sites=min_sites,
        max_sites=max_sites,
        exclude=measured_exclude_set(lpe_data, preset_id, args.exclude_known),
        max_candidates=args.max_candidates,
    )


def combo_preset_id(args) -> int:
    if args.round is not None:
        return args.round
    return 2 if args.training_preset == "single" else 3


def cmd_train_combo(args):
    cfg = load_config(args.config)
    data_path = resolve_path(args.data or cfg.get("lpe_data_pkl", "data/lpe1439.pkl"), repo_root())
    lpe_data = load_or_prepare(cfg, data_path)
    set_seed(args.seed)
    device = get_device(args.force_cpu)
    preset_id = combo_preset_id(args)
    if preset_id == 2:
        train_protein = concatenate_rounds(lpe_data, ["round1"], "LPE1439_round1_for_round2")
    elif preset_id == 3:
        train_protein = concatenate_rounds(lpe_data, ["round1", "round2"], "LPE1439_round1_round2_for_round3")
    else:
        raise ValueError("Combination training preset must resolve to 2 or 3")

    candidate_labels = candidate_labels_from_args(args, lpe_data, train_protein)
    if not candidate_labels:
        raise ValueError("No combination candidates were generated")

    tokenizer = load_tokenizer(**model_args(cfg))
    train_labels, train_sequences, targets = records_from_protein(train_protein)
    candidate_labels, candidate_sequences = records_from_labels(lpe_data["wild_type"], candidate_labels)

    run_name = args.run_name or args.training_preset or f"round{preset_id}"
    out_dir = resolve_path(args.output_dir, repo_root()) / run_name

    def model_factory():
        return load_adapter_model(
            **model_args(cfg),
            adapter_dir=resolve_path(args.checkpoint, repo_root()),
            output_hidden_states=True,
            trainable=True,
        )

    model, head, metadata = fit_full_sequence_model_cv_then_final(
        model_factory,
        tokenizer,
        train_labels,
        train_sequences,
        targets,
        output_dir=out_dir,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.learning_rate,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        folds=args.folds,
        seed=args.seed,
        device=device,
    )
    model.save_pretrained(out_dir / "encoder")
    torch.save(model.state_dict(), out_dir / "encoder" / "encoder_full_state.pt")
    tokenizer.save_pretrained(out_dir / "encoder")
    preds = predict_sequences_with_full_model(
        model,
        head,
        tokenizer,
        candidate_labels,
        candidate_sequences,
        metadata,
        args.embed_batch_size,
        device,
    )
    table = prediction_table(candidate_labels, preds)
    out_csv = out_dir / "combo_predictions.csv"
    table.to_csv(out_csv, index=False)
    table.head(args.top_k).to_csv(out_dir / f"combo_predictions_top{args.top_k}.csv", index=False)
    with (out_dir / "combo_run.json").open("w") as handle:
        json.dump(
            {
                "round": args.round,
                "training_preset": args.training_preset,
                "train_variants": len(train_labels),
                "candidate_variants": len(candidate_labels),
                "exclude_known": args.exclude_known,
                "top_k": args.top_k,
            },
            handle,
            indent=2,
        )
    print(f"Saved {len(candidate_labels)} combo predictions to {out_csv}")


def build_parser():
    parser = argparse.ArgumentParser(prog="primel", description="PRIME-L training and prediction pipeline")
    default_config = str(repo_root() / "configs" / "default.json")
    parser.add_argument("--config", default=default_config)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("prepare", help="Parse This_study Excel workbooks into PRIME-L pkl")
    p.add_argument("--config", default=default_config)
    p.add_argument("--output")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("finetune-ala", help="LoRA/FSFP fine-tuning on LPE1439 alanine scan")
    p.add_argument("--config", default=default_config)
    p.add_argument("--data")
    p.add_argument("--output", default="outputs/checkpoints/primel_ala")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--meta-epochs", type=int, default=10)
    p.add_argument("--train-batch", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--embed-batch-size", type=int, default=1)
    p.add_argument("--list-size", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--optimizer", default="adam")
    p.add_argument("--max-grad-norm", type=float, default=3.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--mask", choices=["train", "eval", "all", "none"], default="none")
    p.add_argument("--lora-r", type=int)
    p.add_argument("--meta-tasks", type=int, default=0)
    p.add_argument("--gemme-data")
    p.add_argument("--max-gemme-records", type=int)
    p.add_argument("--pseudo-task-size", type=int, default=0)
    p.add_argument("--max-aux-records", type=int, default=512)
    p.add_argument("--meta-train-batch", type=int, default=4)
    p.add_argument("--meta-eval-batch", type=int, default=16)
    p.add_argument("--adapt-lr", type=float, default=5e-3)
    p.add_argument("--adapt-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=666666)
    p.add_argument("--force-cpu", action="store_true")
    p.set_defaults(func=cmd_finetune_ala)

    p = sub.add_parser("predict-single", help="Score all possible single-site substitutions")
    p.add_argument("--config", default=default_config)
    p.add_argument("--data")
    p.add_argument("--checkpoint", default="outputs/checkpoints/primel_ala")
    p.add_argument("--output", default="outputs/predictions/single_site_predictions.csv")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--top-k", type=int, default=97)
    p.add_argument("--limit", type=int)
    p.add_argument("--mask", choices=["eval", "all", "none"], default="none")
    p.add_argument("--force-cpu", action="store_true")
    p.set_defaults(func=cmd_predict_single)

    p = sub.add_parser("train-combo", help="Train the multi-site regression head and rank combination candidates")
    p.add_argument("--config", default=default_config)
    p.add_argument("--data")
    p.add_argument("--checkpoint", default="outputs/checkpoints/primel_ala")
    p.add_argument("--round", type=int, choices=[2, 3], help="Backward-compatible preset id; prefer --training-preset")
    p.add_argument("--training-preset", choices=["single", "cumulative"], default="single")
    p.add_argument("--run-name")
    p.add_argument("--single-predictions")
    p.add_argument("--seed-top-k", type=int, default=101)
    p.add_argument("--candidates")
    p.add_argument("--sites", nargs=2, type=int)
    p.add_argument("--max-candidates", type=int, default=200000)
    p.add_argument("--exclude-known", choices=["seen", "all", "none"], default="seen")
    p.add_argument("--output-dir", default="outputs/checkpoints/combo")
    p.add_argument("--embed-batch-size", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=666666)
    p.add_argument("--force-cpu", action="store_true")
    p.set_defaults(func=cmd_train_combo)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
