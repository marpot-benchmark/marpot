
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from training.valid_pixels import match_valid_targets, prepare_mask_targets


def _multi_positive_contrastive(
    sim: torch.Tensor, pos_mask: torch.Tensor
) -> torch.Tensor:
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return sim.new_tensor(0.0)
    a = torch.logsumexp(sim[neg_mask], dim=0)
    b = torch.logsumexp(-sim[pos_mask], dim=0)
    return F.softplus(a + b)


def association_loss_pair(
    key_embed: torch.Tensor,
    ref_embed: torch.Tensor,
    key_ids: torch.Tensor,
    ref_ids: torch.Tensor,
    temperature: float = 0.07,
) -> Optional[torch.Tensor]:
    if key_embed.numel() == 0 or ref_embed.numel() == 0:
        return None

    valid_k = key_ids >= 0
    valid_r = ref_ids >= 0
    if valid_k.sum() == 0 or valid_r.sum() == 0:
        return None
    key_embed, key_ids = key_embed[valid_k], key_ids[valid_k]
    ref_embed, ref_ids = ref_embed[valid_r], ref_ids[valid_r]

    sim = (key_embed @ ref_embed.t()) / max(temperature, 1e-6)
    losses = []
    for i in range(key_embed.shape[0]):
        pos = ref_ids == key_ids[i]
        if pos.any() and (~pos).any():
            losses.append(_multi_positive_contrastive(sim[i], pos))
    if not losses:
        return None
    return torch.stack(losses).mean()


def query_ids_from_matcher(
    matcher,
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    targets: List[dict],
):
    targets = prepare_mask_targets(targets)
    indices = match_valid_targets(matcher, mask_logits, class_logits, targets)

    per_sample = []
    for b, (pred_idx, tgt_idx) in enumerate(indices):
        pseudo = targets[b].get("pseudo_track_ids")
        if pseudo is None:
            ids = torch.full((len(tgt_idx),), -1, dtype=torch.long, device=mask_logits.device)
        else:
            pseudo = pseudo.to(mask_logits.device)
            ids = pseudo[tgt_idx.to(pseudo.device)].long()
        per_sample.append((pred_idx.to(mask_logits.device), ids))
    return per_sample
