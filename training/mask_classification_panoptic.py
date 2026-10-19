# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule


class MaskClassificationPanoptic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        stuff_classes: list[int],
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        load_ckpt_reid_head: bool = False,
        eval_max_size: Optional[int] = None,
        assoc_coefficient: float = 0.0,
        assoc_temperature: float = 0.07,
        trainable_scope: str = "all",
        class_weights: Optional[list[float]] = None,
        foreground_aware_enabled: bool = False,
        foreground_aware_class_ids: Optional[list[int]] = None,
        foreground_aware_max_area: int = 1024,
        foreground_aware_num_points: int = 64,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
            load_ckpt_reid_head=load_ckpt_reid_head,
            eval_max_size=eval_max_size,
            assoc_coefficient=assoc_coefficient,
            assoc_temperature=assoc_temperature,
            trainable_scope=trainable_scope,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = stuff_classes

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
            class_weights=class_weights,
            foreground_aware_enabled=foreground_aware_enabled,
            foreground_aware_class_ids=foreground_aware_class_ids,
            foreground_aware_max_area=foreground_aware_max_area,
            foreground_aware_num_points=foreground_aware_num_points,
        )

        thing_classes = [i for i in range(num_classes) if i not in stuff_classes]
        self.init_metrics_panoptic(
            thing_classes,
            stuff_classes,
            self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1,
        )

    def _cap_eval_size(self, size):
        h, w = int(size[0]), int(size[1])
        if self.eval_max_size is None:
            return (h, w)
        long_side = max(h, w)
        if long_side <= self.eval_max_size:
            return (h, w)
        scale = self.eval_max_size / long_side
        return (max(1, round(h * scale)), max(1, round(w * scale)))

    @staticmethod
    def _resize_per_pixel_target(per_pixel_target, size):
        if tuple(per_pixel_target.shape[:2]) == tuple(size):
            return per_pixel_target
        resized = F.interpolate(
            per_pixel_target.permute(2, 0, 1)[None].float(),
            size=tuple(size),
            mode="nearest",
        )[0]
        return resized.permute(1, 2, 0).to(per_pixel_target.dtype)

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        eval_sizes = [self._cap_eval_size(s) for s in img_sizes]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        outputs = self(transformed_imgs)
        mask_logits_per_layer, class_logits_per_layer = outputs[0], outputs[1]

        is_crowds = [target["is_crowd"] for target in targets]
        targets = self.to_per_pixel_targets_panoptic(targets)
        if self.eval_max_size is not None:
            targets = [
                self._resize_per_pixel_target(target, size)
                for target, size in zip(targets, eval_sizes)
            ]

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, eval_sizes
            )
            preds = self.to_per_pixel_preds_panoptic(
                mask_logits,
                class_logits,
                self.stuff_classes,
                self.mask_thresh,
                self.overlap_thresh,
            )
            self.update_metrics_panoptic(preds, targets, is_crowds, i)

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_panoptic("val", log_per_class=True)

    def on_validation_end(self):
        self._on_eval_end_panoptic("val")

    def test_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end_panoptic("test", log_per_class=True)

    def on_test_end(self):
        self._on_eval_end_panoptic("test")
