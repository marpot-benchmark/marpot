import torch
from torchvision import tv_tensors


def filter_target_instances(target, keep):
    return {
        key: value if key == "valid_mask" else tv_tensors.wrap(value[keep], like=value)
        for key, value in target.items()
    }


def exclude_crowd_pixels(target):
    if "is_crowd" not in target or not bool(target["is_crowd"].any()):
        return target
    target = dict(target)
    crowd = target["masks"][target["is_crowd"].bool()].bool().any(0)
    valid = target.get("valid_mask", torch.ones_like(crowd)).bool() & ~crowd
    target["valid_mask"] = tv_tensors.Mask(valid)
    return target
