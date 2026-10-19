from __future__ import annotations

import torch.nn as nn


def apply_trainable_scope(network: nn.Module, scope: str) -> None:
    if scope not in {"all", "heads", "last4"}:
        raise ValueError("trainable_scope must be one of: all, heads, last4")
    if scope == "all":
        return

    for param in network.parameters():
        param.requires_grad_(False)
    for module_name in ("q", "class_head", "mask_head", "upscale"):
        module = getattr(network, module_name)
        for param in module.parameters():
            param.requires_grad_(True)

    if scope == "last4":
        backbone = network.encoder.backbone
        for block in backbone.blocks[-network.num_blocks :]:
            for param in block.parameters():
                param.requires_grad_(True)
        for param in backbone.norm.parameters():
            param.requires_grad_(True)
