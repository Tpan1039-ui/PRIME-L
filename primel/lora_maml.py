from __future__ import annotations

import torch
from peft.tuners.lora import LoraLayer


LORA_MODULE_ATTRS = ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_dropout"]


def pack_lora_layers(model):
    packed_layers = torch.nn.ModuleList([])
    names = []
    key_list = [key for key, _ in model.named_modules() if "lora" not in key]
    for key in key_list:
        target = model.get_submodule(key)
        if isinstance(target, LoraLayer):
            for attr in LORA_MODULE_ATTRS:
                modules = getattr(target, attr)
                if model.active_adapter in modules:
                    names.append(".".join([key, attr]))
                    packed_layers.append(modules)
    return names, packed_layers


def replace_modules(model, module_names, new_modules):
    for name, module in zip(module_names, new_modules):
        parts = name.split(".")
        parent = model.get_submodule(".".join(parts[:-1]))
        setattr(parent, parts[-1], module)
