# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from training.valid_pixels import match_valid_targets, prepare_mask_targets, sample_valid_coordinates
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
    pair_wise_dice_loss,
    pair_wise_sigmoid_cross_entropy_loss,
    sample_point,
    dice_loss,
    sigmoid_cross_entropy_loss,
)


def _sample_foreground_coordinates(
    mask: torch.Tensor, num_points: int
) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"mask must be [H, W], got shape {tuple(mask.shape)}")
    if num_points <= 0:
        return torch.empty((0, 2), dtype=torch.float32, device=mask.device)

    foreground = torch.nonzero(mask > 0.5, as_tuple=False)
    if foreground.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=mask.device)

    selected = foreground[
        torch.randint(foreground.shape[0], (num_points,), device=mask.device)
    ]
    height, width = mask.shape
    coordinates = torch.stack(
        (
            (selected[:, 1].float() + 0.5) / width,
            (selected[:, 0].float() + 0.5) / height,
        ),
        dim=-1,
    )
    return coordinates


class ForegroundAwareHungarianMatcher(Mask2FormerHungarianMatcher):
    def __init__(
        self,
        *,
        foreground_class_ids: Optional[List[int]],
        foreground_max_area: int,
        foreground_num_points: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if foreground_max_area <= 0:
            raise ValueError("foreground_max_area must be positive")
        if foreground_num_points <= 0:
            raise ValueError("foreground_num_points must be positive")
        self.foreground_class_ids = (
            None
            if foreground_class_ids is None
            else tuple(int(class_id) for class_id in foreground_class_ids)
        )
        self.foreground_max_area = int(foreground_max_area)
        self.foreground_num_points = int(foreground_num_points)

    def _eligible_target_indices(
        self, target_masks: torch.Tensor, target_classes: torch.Tensor
    ) -> List[int]:
        if target_masks.shape[0] == 0:
            return []
        areas = (target_masks > 0.5).flatten(1).sum(1)
        eligible = (areas > 0) & (areas <= self.foreground_max_area)
        if self.foreground_class_ids is not None:
            class_allowed = torch.zeros_like(eligible)
            for class_id in self.foreground_class_ids:
                class_allowed |= target_classes == class_id
            eligible &= class_allowed
        return torch.nonzero(eligible, as_tuple=False).flatten().tolist()

    def _sample_shared_coordinates(
        self, target_masks: torch.Tensor, target_classes: torch.Tensor
    ) -> torch.Tensor:
        coordinates = torch.rand(
            1, self.num_points, 2, device=target_masks.device
        )
        eligible = self._eligible_target_indices(target_masks, target_classes)
        if not eligible:
            return coordinates

        positive_budget = min(
            self.num_points,
            max(len(eligible), self.num_points // 4),
        )
        points_per_target = min(
            self.foreground_num_points,
            max(1, positive_budget // len(eligible)),
        )

        offset = 0
        for target_index in eligible:
            if offset >= positive_budget:
                break
            count = min(points_per_target, positive_budget - offset)
            sampled = _sample_foreground_coordinates(
                target_masks[target_index], count
            )
            coordinates[0, offset : offset + sampled.shape[0]] = sampled
            offset += sampled.shape[0]
        return coordinates

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
    ):
        indices = []
        for batch_index in range(masks_queries_logits.shape[0]):
            pred_probabilities = class_queries_logits[batch_index].softmax(-1)
            pred_masks = masks_queries_logits[batch_index]
            target_classes = class_labels[batch_index]
            target_masks = mask_labels[batch_index].to(pred_masks)

            if target_masks.shape[0] == 0:
                empty = torch.empty(0, dtype=torch.int64)
                indices.append((empty, empty.clone()))
                continue

            cost_class = -pred_probabilities[:, target_classes]
            point_coordinates = self._sample_shared_coordinates(
                target_masks, target_classes
            )

            sampled_targets = sample_point(
                target_masks[:, None],
                point_coordinates.repeat(target_masks.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)
            sampled_predictions = sample_point(
                pred_masks[:, None],
                point_coordinates.repeat(pred_masks.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            cost_mask = pair_wise_sigmoid_cross_entropy_loss(
                sampled_predictions, sampled_targets
            )
            cost_dice = pair_wise_dice_loss(sampled_predictions, sampled_targets)
            cost_matrix = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            cost_matrix = torch.nan_to_num(
                cost_matrix, nan=0.0, posinf=1e10, neginf=-1e10
            ).clamp(min=-1e10, max=1e10)
            pred_indices, target_indices = linear_sum_assignment(
                cost_matrix.float().cpu()
            )
            indices.append(
                (
                    torch.as_tensor(pred_indices, dtype=torch.int64),
                    torch.as_tensor(target_indices, dtype=torch.int64),
                )
            )
        return indices


class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        class_weights: Optional[List[float]] = None,
        foreground_aware_enabled: bool = False,
        foreground_aware_class_ids: Optional[List[int]] = None,
        foreground_aware_max_area: int = 1024,
        foreground_aware_num_points: int = 64,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        self.foreground_aware_enabled = foreground_aware_enabled
        self.foreground_aware_class_ids = (
            None
            if foreground_aware_class_ids is None
            else tuple(int(class_id) for class_id in foreground_aware_class_ids)
        )
        self.foreground_aware_max_area = int(foreground_aware_max_area)
        self.foreground_aware_num_points = int(foreground_aware_num_points)
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        if class_weights is not None:
            if len(class_weights) != self.num_labels:
                raise ValueError(
                    f"class_weights must have {self.num_labels} entries "
                    f"(one per class), got {len(class_weights)}"
                )
            empty_weight[: self.num_labels] = torch.tensor(
                class_weights, dtype=empty_weight.dtype
            )
        self.register_buffer("empty_weight", empty_weight)

        matcher_kwargs = dict(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )
        if foreground_aware_enabled:
            self.matcher = ForegroundAwareHungarianMatcher(
                foreground_class_ids=foreground_aware_class_ids,
                foreground_max_area=foreground_aware_max_area,
                foreground_num_points=foreground_aware_num_points,
                **matcher_kwargs,
            )
        else:
            self.matcher = Mask2FormerHungarianMatcher(**matcher_kwargs)

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
    ):
        targets = prepare_mask_targets(targets)
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        indices = match_valid_targets(
            self.matcher, masks_queries_logits, class_queries_logits, targets
        )
        valid_masks = [target.get("valid_mask") for target in targets]
        loss_masks = self.loss_masks(
            masks_queries_logits, mask_labels, indices, class_labels, valid_masks
        )
        supervised = [b for b, valid in enumerate(valid_masks)
                      if valid is None or bool(valid.any())]
        if len(supervised) == len(targets):
            loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)
        elif supervised:
            loss_classes = self.loss_labels(
                class_queries_logits[supervised], [class_labels[b] for b in supervised],
                [indices[b] for b in supervised],
            )
        else:
            loss_classes = {"loss_cross_entropy": class_queries_logits.sum() * 0.0}

        return {**loss_masks, **loss_classes}

    def loss_masks(
        self, masks_queries_logits, mask_labels, indices, class_labels=None,
        valid_masks=None,
    ):
        if valid_masks is not None and any(v is not None for v in valid_masks):
            loss_masks = self._valid_pixel_loss_masks(
                masks_queries_logits, mask_labels, indices, class_labels, valid_masks
            )
        elif not self.foreground_aware_enabled:
            loss_masks = super().loss_masks(
                masks_queries_logits, mask_labels, indices, 1
            )
        else:
            loss_masks = self._foreground_aware_loss_masks(
                masks_queries_logits, mask_labels, indices, class_labels
            )

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def _valid_pixel_loss_masks(self, logits, labels, indices, classes, valid_masks):
        total = {"loss_mask": logits.sum() * 0.0, "loss_dice": logits.sum() * 0.0}
        for b, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            predictions = logits[b, pred_idx][:, None]
            targets = labels[b][target_idx][:, None]
            valid = valid_masks[b]
            if valid is not None:
                valid = valid.to(device=logits.device, dtype=torch.bool)
            with torch.no_grad():
                if valid is None or bool(valid.all()):
                    coordinates = self.sample_points_using_uncertainty(
                        predictions, self.calculate_uncertainty, self.num_points,
                        self.oversample_ratio, self.importance_sample_ratio,
                    )
                else:
                    n = len(pred_idx)
                    over = int(self.num_points * self.oversample_ratio)
                    candidates = sample_valid_coordinates(valid, n * over).reshape(n, over, 2)
                    values = sample_point(predictions, candidates, align_corners=False)
                    n_hard = int(self.importance_sample_ratio * self.num_points)
                    hard = self.calculate_uncertainty(values)[:, 0].topk(n_hard, dim=1).indices
                    coordinates = candidates.gather(1, hard[..., None].expand(-1, -1, 2))
                    n_random = self.num_points - n_hard
                    if n_random:
                        coordinates = torch.cat((coordinates, sample_valid_coordinates(
                            valid, n * n_random).reshape(n, n_random, 2)), dim=1)
                if self.foreground_aware_enabled:
                    for j, target_id in enumerate(target_idx.tolist()):
                        mask = targets[j, 0] > 0.5
                        if valid is not None:
                            mask = mask & valid
                        area = int(mask.sum())
                        cls = int(classes[b][target_id])
                        allowed = self.foreground_aware_class_ids is None or cls in self.foreground_aware_class_ids
                        if 0 < area <= self.foreground_aware_max_area and allowed:
                            count = min(self.foreground_aware_num_points, self.num_points)
                            coordinates[j, :count] = sample_valid_coordinates(mask, count)
                point_labels = sample_point(targets, coordinates, align_corners=False).squeeze(1)
            point_logits = sample_point(predictions, coordinates, align_corners=False).squeeze(1)
            total["loss_mask"] = total["loss_mask"] + sigmoid_cross_entropy_loss(point_logits, point_labels, 1)
            total["loss_dice"] = total["loss_dice"] + dice_loss(point_logits, point_labels, 1)
        return total

    def _foreground_aware_loss_masks(
        self, masks_queries_logits, mask_labels, indices, class_labels
    ):
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        pred_masks = masks_queries_logits[src_idx]
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        if pred_masks.shape[0] == 0:
            zero = masks_queries_logits.sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}

        matched_classes = torch.cat(
            [
                labels[target_indices]
                for labels, (_, target_indices) in zip(class_labels, indices)
            ]
        )
        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            areas = (target_masks[:, 0] > 0.5).flatten(1).sum(1)
            eligible = (areas > 0) & (areas <= self.foreground_aware_max_area)
            if self.foreground_aware_class_ids is not None:
                class_allowed = torch.zeros_like(eligible)
                for class_id in self.foreground_aware_class_ids:
                    class_allowed |= matched_classes == class_id
                eligible &= class_allowed
            eligible_indices = torch.nonzero(
                eligible, as_tuple=False
            ).flatten().tolist()
            for index in eligible_indices:
                count = min(self.foreground_aware_num_points, self.num_points)
                point_coordinates[index, :count] = _sample_foreground_coordinates(
                    target_masks[index, 0], count
                )

            point_labels = sample_point(
                target_masks, point_coordinates, align_corners=False
            ).squeeze(1)

        point_logits = sample_point(
            pred_masks, point_coordinates, align_corners=False
        ).squeeze(1)
        return {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, 1),
            "loss_dice": dice_loss(point_logits, point_labels, 1),
        }

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total
