

import json
import random
import re
from pathlib import Path
from typing import Optional, Union, Callable

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import TileExpertTransforms, Transforms
from datasets.target_fields import filter_target_instances
from tile_geometry import sliding_tiles

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3}


def parse_seq_frame(file_name: str) -> tuple[str, int]:
    path = Path(file_name)
    seq_id = path.parent.name
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse frame number from {file_name}")
    return seq_id, int(match.group(1))


class SeaPanopticDataset(TorchDataset):
    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        transforms: Optional[Callable] = None,
        check_empty_targets: bool = True,
        sequence_mode: bool = False,
        ref_frame_max_gap: int = 5,
        ref_max_sample_gap: int = 3,
        pseudo_tracks_path: Optional[str] = None,
        use_gt_track_ids: bool = False,
        copy_paste_enabled: bool = False,
        copy_paste_prob: float = 0.5,
        copy_paste_max_instances: int = 3,
        copy_paste_small_only: bool = False,
        copy_paste_small_max_area: int = 9216,
        copy_paste_class_balanced: bool = False,
        copy_paste_scale_range: tuple[float, float] = (1.0, 1.0),
        copy_paste_random_position: bool = False,
        small_object_crop_enabled: bool = False,
        small_object_crop_prob: float = 0.5,
        small_object_crop_size: int = 1120,
        small_object_max_area: int = 9216,
        small_object_crop_min_visible_ratio: float = 0.0,
        small_object_crop_truncated_policy: str = "keep",
        small_object_crop_apply_to_ref: bool = False,
        tile_expert_enabled: bool = False,
        tile_expert_size: tuple[int, int] = (1120, 1120),
        tile_expert_overlap: float = 0.2,
        tile_expert_positive_prob: float = 0.75,
        tile_expert_max_area: int = 1024,
        tile_expert_min_visible_ratio: float = 0.99,
    ):
        root = Path(root)
        self.img_root = root / "images"
        self.png_root = root / "panoptic"
        self.transforms = transforms

        self.copy_paste_enabled = copy_paste_enabled
        self.copy_paste_prob = copy_paste_prob
        self.copy_paste_max_instances = copy_paste_max_instances
        self.copy_paste_small_only = copy_paste_small_only
        self.copy_paste_small_max_area = copy_paste_small_max_area
        self.copy_paste_class_balanced = copy_paste_class_balanced
        self.copy_paste_scale_range = tuple(copy_paste_scale_range)
        self.copy_paste_random_position = copy_paste_random_position
        self.small_object_crop_enabled = small_object_crop_enabled
        self.small_object_crop_prob = small_object_crop_prob
        self.small_object_crop_size = small_object_crop_size
        self.small_object_max_area = small_object_max_area
        self.small_object_crop_min_visible_ratio = small_object_crop_min_visible_ratio
        self.small_object_crop_truncated_policy = small_object_crop_truncated_policy
        self.small_object_crop_apply_to_ref = small_object_crop_apply_to_ref
        self.tile_expert_enabled = tile_expert_enabled
        self.tile_expert_size = tuple(tile_expert_size)
        self.tile_expert_overlap = tile_expert_overlap
        self.tile_expert_positive_prob = tile_expert_positive_prob
        self.tile_expert_max_area = tile_expert_max_area
        self.tile_expert_min_visible_ratio = tile_expert_min_visible_ratio
        if not 0 <= self.small_object_crop_min_visible_ratio <= 1:
            raise ValueError("small_object_crop_min_visible_ratio must be in [0, 1]")
        if self.small_object_crop_truncated_policy not in ("keep", "drop", "ignore"):
            raise ValueError("small_object_crop_truncated_policy must be keep/drop/ignore")
        if self.tile_expert_enabled and self.small_object_crop_enabled:
            raise ValueError(
                "tile_expert_enabled and small_object_crop_enabled are mutually exclusive"
            )
        if len(self.tile_expert_size) != 2 or min(self.tile_expert_size) <= 0:
            raise ValueError("tile_expert_size must contain two positive integers")
        if not 0 <= self.tile_expert_overlap < 1:
            raise ValueError("tile_expert_overlap must be in [0, 1)")
        if not 0 <= self.tile_expert_positive_prob <= 1:
            raise ValueError("tile_expert_positive_prob must be in [0, 1]")
        if self.tile_expert_max_area <= 0:
            raise ValueError("tile_expert_max_area must be positive")
        if not 0 <= self.tile_expert_min_visible_ratio <= 1:
            raise ValueError("tile_expert_min_visible_ratio must be in [0, 1]")
        if len(self.copy_paste_scale_range) != 2 or not (
            0 < self.copy_paste_scale_range[0] <= self.copy_paste_scale_range[1]
        ):
            raise ValueError("copy_paste_scale_range must be two positive ordered values")

        self.sequence_mode = sequence_mode
        self.ref_frame_max_gap = ref_frame_max_gap
        self.ref_max_sample_gap = ref_max_sample_gap

        self.use_gt_track_ids = use_gt_track_ids

        self.pseudo_tracks = None
        if pseudo_tracks_path is not None:
            with open(pseudo_tracks_path) as pf:
                self.pseudo_tracks = json.load(pf)

        with open(root / "annotations" / f"panoptic_{split}.json") as f:
            annotation_data = json.load(f)

        image_id_to_file_name = {
            image["id"]: image["file_name"] for image in annotation_data["images"]
        }

        self.samples = []
        for annotation in annotation_data["annotations"]:
            segments = [
                (
                    segment_info["id"],
                    segment_info["category_id"],
                    bool(segment_info["iscrowd"]),
                    int(segment_info.get("track_id", -1)),
                )
                for segment_info in annotation["segments_info"]
                if segment_info["category_id"] in CLASS_MAPPING
            ]

            if check_empty_targets and not segments:
                continue

            self.samples.append(
                (
                    image_id_to_file_name[annotation["image_id"]],
                    annotation["file_name"],
                    segments,
                )
            )

        if self.sequence_mode:
            self._build_sequence_index()

        if self.copy_paste_enabled:
            self.thing_indices = [
                idx
                for idx, (_, _, segs) in enumerate(self.samples)
                if any(cat in (1, 2) and not crowd for (_, cat, crowd, _) in segs)
            ]
            self.thing_indices_by_class = {
                CLASS_MAPPING[cat]: [
                    idx
                    for idx, (_, _, segs) in enumerate(self.samples)
                    if any(c == cat and not crowd for (_, c, crowd, _) in segs)
                ]
                for cat in (1, 2)
            }

    def _build_sequence_index(self):
        self.sample_seq = []
        self.sample_frame = []
        seq_to_items = {}
        for idx, (img_file, _, _) in enumerate(self.samples):
            seq_id, frame_no = parse_seq_frame(img_file)
            self.sample_seq.append(seq_id)
            self.sample_frame.append(frame_no)
            seq_to_items.setdefault(seq_id, []).append(idx)

        self.seq_to_positions = {}
        self.position_in_seq = [0] * len(self.samples)
        for seq_id, idxs in seq_to_items.items():
            idxs.sort(key=lambda i: self.sample_frame[i])
            self.seq_to_positions[seq_id] = idxs
            for pos, i in enumerate(idxs):
                self.position_in_seq[i] = pos

    def _sample_ref_index(self, index: int) -> int:
        seq_id = self.sample_seq[index]
        positions = self.seq_to_positions[seq_id]
        pos = self.position_in_seq[index]
        frame_no = self.sample_frame[index]

        candidates = []
        for dp in range(-self.ref_max_sample_gap, self.ref_max_sample_gap + 1):
            if dp == 0:
                continue
            j = pos + dp
            if j < 0 or j >= len(positions):
                continue
            cand_idx = positions[j]
            if abs(self.sample_frame[cand_idx] - frame_no) <= self.ref_frame_max_gap:
                candidates.append(cand_idx)

        return random.choice(candidates) if candidates else index

    def __len__(self):
        return len(self.samples)

    def _build_sample(self, index: int):
        img_file, png_file, segments = self.samples[index]

        img = tv_tensors.Image(Image.open(self.img_root / img_file).convert("RGB"))
        png = tv_tensors.Mask(
            Image.open(self.png_root / png_file), dtype=torch.long
        )

        ids = png[0, :, :] + png[1, :, :] * 256 + png[2, :, :] * 256**2

        if ids.shape[-2:] != img.shape[-2:]:
            ids = F.resize(
                ids[None],
                list(img.shape[-2:]),
                interpolation=F.InterpolationMode.NEAREST,
            )[0]

        masks, labels, is_crowd, seg_ids, track_ids = [], [], [], [], []
        valid_mask = torch.zeros_like(ids, dtype=torch.bool)
        for seg_id, cat_id, crowd, track_id in segments:
            mask = ids == seg_id
            if not mask.any():
                continue
            masks.append(mask)
            if not crowd:
                valid_mask |= mask
            labels.append(CLASS_MAPPING[cat_id])
            is_crowd.append(crowd)
            seg_ids.append(seg_id)
            track_ids.append(track_id)

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks) if masks else torch.zeros(
                (0, *ids.shape), dtype=torch.bool)),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
        }

        if not bool(valid_mask.all()):
            target["valid_mask"] = tv_tensors.Mask(valid_mask)

        if self.sequence_mode:
            target["seg_ids"] = torch.tensor(seg_ids, dtype=torch.long)

            if self.use_gt_track_ids:
                target["pseudo_track_ids"] = torch.tensor(track_ids, dtype=torch.long)
            elif self.pseudo_tracks is not None:
                seq_id, frame_no = parse_seq_frame(img_file)
                lut = self.pseudo_tracks.get(seq_id, {}).get(str(frame_no), {})
                pseudo = [int(lut.get(str(s), -1)) for s in seg_ids]
                target["pseudo_track_ids"] = torch.tensor(pseudo, dtype=torch.long)

        return img, target

    def _copy_paste(self, img, target):
        if not self.thing_indices or random.random() >= self.copy_paste_prob:
            return img, target

        preferred_class = None
        if self.copy_paste_class_balanced:
            available = [c for c, indices in self.thing_indices_by_class.items() if indices]
            preferred_class = random.choice(available)
            src_index = random.choice(self.thing_indices_by_class[preferred_class])
        else:
            src_index = random.choice(self.thing_indices)
        src_img, src_target = self._build_sample(src_index)

        src_labels = src_target["labels"]
        src_crowd = src_target["is_crowd"].bool()
        is_thing = (src_labels == 0) | (src_labels == 1)
        sel = torch.nonzero(is_thing & (~src_crowd)).flatten().tolist()
        if preferred_class is not None:
            sel = [k for k in sel if int(src_labels[k]) == preferred_class]
        if self.copy_paste_small_only:
            src_areas = src_target["masks"].flatten(1).sum(1)
            sel = [k for k in sel if src_areas[k] <= self.copy_paste_small_max_area]
        if not sel:
            return img, target
        if self.copy_paste_class_balanced:
            by_class = {0: [], 1: []}
            for k in sel:
                by_class[int(src_labels[k])].append(k)
            sel = []
            while len(sel) < self.copy_paste_max_instances and any(by_class.values()):
                available = [c for c, items in by_class.items() if items]
                c = random.choice(available)
                sel.append(by_class[c].pop(random.randrange(len(by_class[c]))))
        else:
            random.shuffle(sel)
            sel = sel[: self.copy_paste_max_instances]

        tgt_h, tgt_w = int(img.shape[-2]), int(img.shape[-1])
        if tuple(src_img.shape[-2:]) != (tgt_h, tgt_w):
            src_img_t = F.resize(
                src_img.as_subclass(torch.Tensor),
                [tgt_h, tgt_w],
                interpolation=F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            src_masks_t = (
                F.resize(
                    src_target["masks"].as_subclass(torch.Tensor).to(torch.uint8),
                    [tgt_h, tgt_w],
                    interpolation=F.InterpolationMode.NEAREST,
                )
                > 0
            )
        else:
            src_img_t = src_img.as_subclass(torch.Tensor)
            src_masks_t = src_target["masks"].as_subclass(torch.Tensor).bool()

        img_t = img.as_subclass(torch.Tensor).clone()
        new_masks, new_labels = [], []
        paste_union = torch.zeros((tgt_h, tgt_w), dtype=torch.bool)
        for k in sel:
            m = src_masks_t[k]
            if not m.any():
                continue
            enhanced_geometry = self.copy_paste_random_position or self.copy_paste_scale_range != (1.0, 1.0)
            if enhanced_geometry:
                ys, xs = torch.where(m)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                patch_img, patch_mask = src_img_t[:, y0:y1, x0:x1], m[y0:y1, x0:x1]
                scale = random.uniform(*self.copy_paste_scale_range)
                ph = max(1, min(tgt_h, round(patch_mask.shape[-2] * scale)))
                pw = max(1, min(tgt_w, round(patch_mask.shape[-1] * scale)))
                patch_img = F.resize(patch_img, [ph, pw], antialias=True)
                patch_mask = F.resize(
                    patch_mask[None].to(torch.uint8), [ph, pw],
                    interpolation=F.InterpolationMode.NEAREST,
                )[0].bool()
                if self.copy_paste_random_position:
                    py = random.randint(0, tgt_h - ph)
                    px = random.randint(0, tgt_w - pw)
                else:
                    py, px = min(y0, tgt_h - ph), min(x0, tgt_w - pw)
                m = torch.zeros_like(m)
                m[py:py + ph, px:px + pw] = patch_mask
                placed_img = torch.zeros_like(src_img_t)
                placed_img[:, py:py + ph, px:px + pw] = patch_img
            else:
                placed_img = src_img_t
            for j in range(len(new_masks)):
                new_masks[j] = new_masks[j] & (~m)
            img_t = torch.where(m, placed_img, img_t)
            new_masks.append(m.clone())
            new_labels.append(int(src_labels[k].item()))
            paste_union |= m

        if not new_masks:
            return img, target

        old_masks = target["masks"].as_subclass(torch.Tensor).bool() & (~paste_union)[None]
        keep_old = old_masks.flatten(1).any(1)

        new_masks_t = torch.stack(new_masks)
        n_new = new_masks_t.shape[0]

        def _cat_new(old_val, fill_value):
            return torch.cat(
                [
                    old_val[keep_old],
                    torch.full((n_new,), fill_value, dtype=old_val.dtype),
                ]
            )

        target = dict(target)
        target["masks"] = tv_tensors.Mask(
            torch.cat([old_masks[keep_old], new_masks_t])
        )
        target["labels"] = torch.cat(
            [target["labels"][keep_old], torch.tensor(new_labels, dtype=target["labels"].dtype)]
        )
        target["is_crowd"] = _cat_new(target["is_crowd"], False)
        if "valid_mask" in target:
            target["valid_mask"] = tv_tensors.Mask(target["valid_mask"].bool() | paste_union)
        if "seg_ids" in target:
            target["seg_ids"] = _cat_new(target["seg_ids"], -1)
        if "pseudo_track_ids" in target:
            target["pseudo_track_ids"] = _cat_new(target["pseudo_track_ids"], -1)

        return tv_tensors.Image(img_t), target

    def _sample_small_object_crop_window(self, img, target):
        if random.random() >= self.small_object_crop_prob:
            return None

        masks = target["masks"].as_subclass(torch.Tensor).bool()
        labels = target["labels"]
        areas = masks.flatten(1).sum(1)
        candidates = torch.nonzero(
            ((labels == 0) | (labels == 1))
            & (~target["is_crowd"].bool())
            & (areas > 0)
            & (areas <= self.small_object_max_area)
        ).flatten().tolist()
        if not candidates:
            return None

        selected = random.choice(candidates)
        ys, xs = torch.where(masks[selected])
        center_y = int((int(ys.min()) + int(ys.max())) / 2)
        center_x = int((int(xs.min()) + int(xs.max())) / 2)
        h, w = int(img.shape[-2]), int(img.shape[-1])
        crop_h, crop_w = min(self.small_object_crop_size, h), min(self.small_object_crop_size, w)

        jitter_y = random.randint(-crop_h // 4, crop_h // 4)
        jitter_x = random.randint(-crop_w // 4, crop_w // 4)
        top = max(0, min(h - crop_h, center_y + jitter_y - crop_h // 2))
        left = max(0, min(w - crop_w, center_x + jitter_x - crop_w // 2))

        return top, left, crop_h, crop_w

    def _apply_crop_window(self, img, target, window):
        img_orig, target_orig = img, target
        top, left, crop_h, crop_w = window
        h, w = int(img.shape[-2]), int(img.shape[-1])
        crop_h, crop_w = min(crop_h, h), min(crop_w, w)
        top, left = min(top, h - crop_h), min(left, w - crop_w)
        original_areas = target["masks"].flatten(1).sum(1).clamp_min(1)
        img = F.crop(img, top, left, crop_h, crop_w)
        target = dict(target)
        target["masks"] = tv_tensors.Mask(
            F.crop(target["masks"], top, left, crop_h, crop_w)
        )
        if "valid_mask" in target:
            target["valid_mask"] = tv_tensors.Mask(
                F.crop(target["valid_mask"], top, left, crop_h, crop_w)
            )
        visible_areas = target["masks"].flatten(1).sum(1)
        valid = visible_areas > 0
        truncated = (visible_areas / original_areas) < self.small_object_crop_min_visible_ratio
        if self.small_object_crop_truncated_policy == "drop":
            dropped_pixels = target["masks"][truncated & valid].bool().any(0)
            if bool(dropped_pixels.any()):
                target["valid_mask"] = tv_tensors.Mask(
                    target.get("valid_mask", torch.ones_like(dropped_pixels)).bool()
                    & ~dropped_pixels
                )
            valid &= ~truncated
        elif self.small_object_crop_truncated_policy == "ignore":
            target["is_crowd"] = target["is_crowd"].clone()
            target["is_crowd"][truncated & valid] = True
        if not valid.any():
            return img_orig, target_orig
        target = filter_target_instances(target, valid)
        return tv_tensors.Image(img), target

    def _small_object_crop(self, img, target):
        window = self._sample_small_object_crop_window(img, target)
        return (img, target) if window is None else self._apply_crop_window(img, target, window)

    def _tile_expert_windows(self, img, target):
        h, w = int(img.shape[-2]), int(img.shape[-1])
        windows = sliding_tiles(h, w, self.tile_expert_size, self.tile_expert_overlap)
        masks = target["masks"].as_subclass(torch.Tensor).bool()
        labels = target["labels"]
        areas = masks.flatten(1).sum(1)
        eligible = torch.nonzero(
            ((labels == 0) | (labels == 1))
            & (~target["is_crowd"].bool())
            & (areas > 0)
            & (areas < self.tile_expert_max_area)
        ).flatten().tolist()

        positive = []
        for window in windows:
            y0, x0, y1, x1 = window
            for index in eligible:
                visible = masks[index, y0:y1, x0:x1].sum()
                if float(visible / areas[index]) >= self.tile_expert_min_visible_ratio:
                    positive.append(window)
                    break
        positive_set = set(positive)
        negative = [window for window in windows if window not in positive_set]
        return windows, positive, negative

    def _sample_tile_expert_window(self, img, target):
        windows, positive, negative = self._tile_expert_windows(img, target)
        if positive and random.random() < self.tile_expert_positive_prob:
            return random.choice(positive)
        return random.choice(negative or windows)

    @staticmethod
    def _apply_tile_expert_window(img, target, window):
        y0, x0, y1, x1 = window
        crop_h, crop_w = y1 - y0, x1 - x0
        img = F.crop(img, y0, x0, crop_h, crop_w)
        target = dict(target)
        target["masks"] = tv_tensors.Mask(
            F.crop(target["masks"], y0, x0, crop_h, crop_w)
        )
        if "valid_mask" in target:
            target["valid_mask"] = tv_tensors.Mask(
                F.crop(target["valid_mask"], y0, x0, crop_h, crop_w)
            )
        valid = target["masks"].flatten(1).any(1)
        target = filter_target_instances(target, valid)
        return tv_tensors.Image(img), target

    def __getitem__(self, index: int):
        img, target = self._build_sample(index)

        crop_window = None
        if self.tile_expert_enabled and self.transforms is not None:
            tile_window = self._sample_tile_expert_window(img, target)
            img, target = self._apply_tile_expert_window(img, target, tile_window)

        if self.small_object_crop_enabled and self.transforms is not None:
            crop_window = self._sample_small_object_crop_window(img, target)
            if crop_window is not None:
                img, target = self._apply_crop_window(img, target, crop_window)

        if self.copy_paste_enabled and self.transforms is not None:
            img, target = self._copy_paste(img, target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.sequence_mode:
            ref_index = self._sample_ref_index(index)
            img_ref, target_ref = self._build_sample(ref_index)
            if crop_window is not None and self.small_object_crop_apply_to_ref:
                img_ref, target_ref = self._apply_crop_window(img_ref, target_ref, crop_window)
            if self.transforms is not None:
                img_ref, target_ref = self.transforms(img_ref, target_ref)
            target["img_ref"] = img_ref
            target["target_ref"] = target_ref
            target["seq_id"] = self.sample_seq[index]
            target["frame_no"] = self.sample_frame[index]
            target["ref_frame_no"] = self.sample_frame[ref_index]
            target["reference_is_distinct"] = ref_index != index
            target["ref_file_name"] = self.samples[ref_index][0]

        return img, target


class SeaPanoptic(LightningDataModule):
    def __init__(
        self,
        path,
        stuff_classes: list[int],
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 4,
        color_jitter_enabled=False,
        scale_range=(0.1, 2.0),
        check_empty_targets=True,
        sequence_mode: bool = False,
        ref_frame_max_gap: int = 5,
        ref_max_sample_gap: int = 3,
        pseudo_tracks_path: Optional[str] = None,
        use_gt_track_ids: bool = False,
        copy_paste_enabled: bool = False,
        copy_paste_prob: float = 0.5,
        copy_paste_max_instances: int = 3,
        copy_paste_small_only: bool = False,
        copy_paste_small_max_area: int = 9216,
        copy_paste_class_balanced: bool = False,
        copy_paste_scale_range: tuple[float, float] = (1.0, 1.0),
        copy_paste_random_position: bool = False,
        small_object_crop_enabled: bool = False,
        small_object_crop_prob: float = 0.5,
        small_object_crop_size: int = 1120,
        small_object_max_area: int = 9216,
        small_object_crop_min_visible_ratio: float = 0.0,
        small_object_crop_truncated_policy: str = "keep",
        small_object_crop_apply_to_ref: bool = False,
        tile_expert_enabled: bool = False,
        tile_expert_size: tuple[int, int] = (1120, 1120),
        tile_expert_overlap: float = 0.2,
        tile_expert_positive_prob: float = 0.75,
        tile_expert_max_area: int = 1024,
        tile_expert_min_visible_ratio: float = 0.99,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.sequence_mode = sequence_mode
        self.ref_frame_max_gap = ref_frame_max_gap
        self.ref_max_sample_gap = ref_max_sample_gap
        self.pseudo_tracks_path = pseudo_tracks_path
        self.use_gt_track_ids = use_gt_track_ids

        self.copy_paste_enabled = copy_paste_enabled
        self.copy_paste_prob = copy_paste_prob
        self.copy_paste_max_instances = copy_paste_max_instances
        self.copy_paste_small_only = copy_paste_small_only
        self.copy_paste_small_max_area = copy_paste_small_max_area
        self.copy_paste_class_balanced = copy_paste_class_balanced
        self.copy_paste_scale_range = copy_paste_scale_range
        self.copy_paste_random_position = copy_paste_random_position
        self.small_object_crop_enabled = small_object_crop_enabled
        self.small_object_crop_prob = small_object_crop_prob
        self.small_object_crop_size = small_object_crop_size
        self.small_object_max_area = small_object_max_area
        self.small_object_crop_min_visible_ratio = small_object_crop_min_visible_ratio
        self.small_object_crop_truncated_policy = small_object_crop_truncated_policy
        self.small_object_crop_apply_to_ref = small_object_crop_apply_to_ref
        self.tile_expert_enabled = tile_expert_enabled
        self.tile_expert_size = tile_expert_size
        self.tile_expert_overlap = tile_expert_overlap
        self.tile_expert_positive_prob = tile_expert_positive_prob
        self.tile_expert_max_area = tile_expert_max_area
        self.tile_expert_min_visible_ratio = tile_expert_min_visible_ratio

        if self.tile_expert_enabled:
            self.transforms = TileExpertTransforms(img_size=img_size)
        else:
            self.transforms = Transforms(
                img_size=img_size,
                color_jitter_enabled=color_jitter_enabled,
                scale_range=scale_range,
            )

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        self.train_dataset = SeaPanopticDataset(
            self.path,
            split="train",
            transforms=self.transforms,
            check_empty_targets=self.check_empty_targets,
            sequence_mode=self.sequence_mode,
            ref_frame_max_gap=self.ref_frame_max_gap,
            ref_max_sample_gap=self.ref_max_sample_gap,
            pseudo_tracks_path=self.pseudo_tracks_path,
            use_gt_track_ids=self.use_gt_track_ids,
            copy_paste_enabled=self.copy_paste_enabled,
            copy_paste_prob=self.copy_paste_prob,
            copy_paste_max_instances=self.copy_paste_max_instances,
            copy_paste_small_only=self.copy_paste_small_only,
            copy_paste_small_max_area=self.copy_paste_small_max_area,
            copy_paste_class_balanced=self.copy_paste_class_balanced,
            copy_paste_scale_range=self.copy_paste_scale_range,
            copy_paste_random_position=self.copy_paste_random_position,
            small_object_crop_enabled=self.small_object_crop_enabled,
            small_object_crop_prob=self.small_object_crop_prob,
            small_object_crop_size=self.small_object_crop_size,
            small_object_max_area=self.small_object_max_area,
            small_object_crop_min_visible_ratio=self.small_object_crop_min_visible_ratio,
            small_object_crop_truncated_policy=self.small_object_crop_truncated_policy,
            small_object_crop_apply_to_ref=self.small_object_crop_apply_to_ref,
            tile_expert_enabled=self.tile_expert_enabled,
            tile_expert_size=self.tile_expert_size,
            tile_expert_overlap=self.tile_expert_overlap,
            tile_expert_positive_prob=self.tile_expert_positive_prob,
            tile_expert_max_area=self.tile_expert_max_area,
            tile_expert_min_visible_ratio=self.tile_expert_min_visible_ratio,
        )
        self.val_dataset = SeaPanopticDataset(
            self.path,
            split="val",
            transforms=None,
            check_empty_targets=self.check_empty_targets,
        )
        self.test_dataset = SeaPanopticDataset(
            self.path,
            split="test",
            transforms=None,
            check_empty_targets=self.check_empty_targets,
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
