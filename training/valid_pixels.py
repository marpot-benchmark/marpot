import torch
from scipy.optimize import linear_sum_assignment
from transformers.models.mask2former.modeling_mask2former import (
    sample_point, pair_wise_dice_loss, pair_wise_sigmoid_cross_entropy_loss,
)


def sample_valid_coordinates(valid, num_points):
    pixels = torch.nonzero(valid, as_tuple=False)
    if pixels.shape[0] == 0:
        raise ValueError("cannot sample an entirely ignored image")
    selected = pixels[torch.randint(len(pixels), (num_points,), device=valid.device)]
    h, w = valid.shape
    return torch.stack(((selected[:, 1].float() + 0.5) / w,
                        (selected[:, 0].float() + 0.5) / h), -1)


def prepare_mask_targets(targets):
    prepared = []
    for target in targets:
        if "valid_mask" not in target or not bool(target["is_crowd"].any()):
            prepared.append(target)
            continue
        keep = ~target["is_crowd"].bool()
        item = dict(target)
        for key in ("masks", "labels", "is_crowd", "seg_ids", "pseudo_track_ids"):
            if key in item:
                item[key] = item[key][keep]
        item["valid_mask"] = target["valid_mask"].bool() & ~target["masks"][~keep].bool().any(0)
        prepared.append(item)
    return prepared


@torch.no_grad()
def match_valid_targets(matcher, masks, classes, targets):
    mask_labels = [t["masks"].to(masks.dtype) for t in targets]
    class_labels = [t["labels"].long() for t in targets]
    if not any("valid_mask" in t for t in targets):
        return matcher(masks_queries_logits=masks, class_queries_logits=classes,
                       mask_labels=mask_labels, class_labels=class_labels)
    result = []
    for b, target in enumerate(targets):
        valid = target.get("valid_mask")
        if valid is None or bool(valid.all()):
            result.extend(matcher(masks_queries_logits=masks[b:b+1],
                                  class_queries_logits=classes[b:b+1],
                                  mask_labels=[mask_labels[b]], class_labels=[class_labels[b]]))
            continue
        valid = valid.to(device=masks.device, dtype=torch.bool)
        if mask_labels[b].shape[0] == 0 or not bool(valid.any()):
            empty = torch.empty(0, dtype=torch.int64)
            result.append((empty, empty.clone()))
            continue
        coordinates = sample_valid_coordinates(valid, matcher.num_points)[None]
        if hasattr(matcher, "_eligible_target_indices"):
            eligible = matcher._eligible_target_indices(mask_labels[b], class_labels[b])
            budget = min(matcher.num_points, max(len(eligible), matcher.num_points // 4))
            per_target = min(matcher.foreground_num_points, max(1, budget // max(len(eligible), 1)))
            offset = 0
            for index in eligible:
                count = min(per_target, budget - offset)
                if count <= 0:
                    break
                foreground = (mask_labels[b][index] > 0.5) & valid
                if bool(foreground.any()):
                    coordinates[0, offset:offset+count] = sample_valid_coordinates(foreground, count)
                    offset += count
        with torch.autocast(device_type=masks.device.type, enabled=False):
            logits = sample_point(masks[b, :, None].float(), coordinates.repeat(masks.shape[1], 1, 1),
                                  align_corners=False).squeeze(1)
            labels = sample_point(mask_labels[b][:, None].float(),
                                  coordinates.repeat(len(mask_labels[b]), 1, 1),
                                  align_corners=False).squeeze(1)
            cost = (matcher.cost_class * -classes[b].float().softmax(-1)[:, class_labels[b]]
                    + matcher.cost_mask * pair_wise_sigmoid_cross_entropy_loss(logits, labels)
                    + matcher.cost_dice * pair_wise_dice_loss(logits, labels))
        if not bool(torch.isfinite(cost).all()):
            raise FloatingPointError("nonfinite matching costs on valid pixels")
        row, col = linear_sum_assignment(cost.cpu())
        result.append((torch.as_tensor(row, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)))
    return result
