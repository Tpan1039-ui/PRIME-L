from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


def load_tokenizer(model_name: str, trust_remote_code: bool = True, local_files_only: bool = False):
    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )


def load_base_model(
    model_name: str,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
    output_hidden_states: bool = False,
):
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
        "output_hidden_states": output_hidden_states,
    }
    if "prime_690m" in model_name.lower():
        model = AutoModel.from_pretrained(model_name, **kwargs)
    else:
        try:
            model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs)
        except Exception:
            model = AutoModel.from_pretrained(model_name, **kwargs)
    if hasattr(model, "config"):
        model.config.output_hidden_states = output_hidden_states
    for name, param in model.named_parameters():
        if "contact_head.regression" in name:
            param.requires_grad = False
    return model


def load_model_and_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
    output_hidden_states: bool = False,
):
    tokenizer = load_tokenizer(model_name, trust_remote_code, local_files_only)
    model = load_base_model(model_name, trust_remote_code, local_files_only, output_hidden_states)
    return model, tokenizer


def apply_lora(model, r: int, alpha: int | None, dropout: float, target_modules: list[str]):
    if r <= 0:
        return model
    config = LoraConfig(
        r=r,
        lora_alpha=alpha or r,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )
    return get_peft_model(model, config)


def load_adapter_model(
    model_name: str,
    adapter_dir: str | Path,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
    output_hidden_states: bool = False,
    trainable: bool = False,
):
    base = load_base_model(model_name, trust_remote_code, local_files_only, output_hidden_states)
    return PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=trainable)


def forward_logits(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    output = model(**batch)
    logits = getattr(output, "logits", None)
    if logits is None:
        raise RuntimeError("Model output does not expose logits; cannot compute mutational scores.")
    return logits


def forward_last_hidden(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    output = model(**batch, output_hidden_states=True)
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        hidden_states = getattr(output, "sequence_hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model output does not expose hidden states.")
    if isinstance(hidden_states, (tuple, list)):
        return hidden_states[-1]
    return hidden_states


def count_trainable(model) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total
