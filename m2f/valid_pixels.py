import torch
import torch.nn.functional as F


def target_valid(target):
    if '_valid_mask' in target:
        return target['_valid_mask']
    masks = target['masks']
    target['_valid_mask'] = masks.bool().any(dim=0)
    return target['_valid_mask']


def repair_coords(coords, valid):
    if coords.numel() == 0:
        return coords
    valid = valid.bool()
    if valid.ndim != 2:
        raise ValueError('valid region must be H x W')
    if bool(valid.all()):
        return coords
    h, w = valid.shape
    x = coords[..., 0] * w - .5
    y = coords[..., 1] * h - .5
    x0 = x.floor().long(); y0 = y.floor().long()
    x1 = x0 + 1; y1 = y0 + 1
    ok = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    for xx, yy in ((x0,y0),(x0,y1),(x1,y0),(x1,y1)):
        ok &= valid[yy.clamp(0,h-1),xx.clamp(0,w-1)]
    if bool(ok.all()):
        return coords
    available = valid.flatten().nonzero(as_tuple=False).flatten()
    if not available.numel():
        if coords.shape[0] == 0:
            return coords
        raise ValueError('no valid spatial supervision in crop')
    ids = available[torch.randint(available.numel(), (int((~ok).sum()),), device=coords.device)]
    replacement = torch.stack(((ids % w + .5) / w, (ids // w + .5) / h),dim=-1).to(coords)
    repaired = coords.clone(); repaired[~ok] = replacement
    return repaired


def repair_matched_coords(coords, targets, indices, canvas_shape=None):
    repaired=[]; offset=0
    for target, (_, matched) in zip(targets, indices):
        count=len(matched)
        valid=target_valid(target)
        if canvas_shape is not None and tuple(valid.shape) != tuple(canvas_shape):
            h,w=canvas_shape
            valid=F.pad(valid,(0,w-valid.shape[-1],0,h-valid.shape[-2]),value=False)
        repaired.append(repair_coords(coords[offset:offset+count],valid))
        offset+=count
    if offset != coords.shape[0]:
        raise ValueError('matched mask order disagrees with point rows')
    return torch.cat(repaired) if repaired else coords


def select_targets(target, selected):
    valid=target_valid(target)
    return {**{k:v[selected] for k,v in target.items() if k!='_valid_mask'},'_valid_mask':valid}
