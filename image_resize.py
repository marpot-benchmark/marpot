from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def resize_uint8_image_pil(
    image: torch.Tensor, size: tuple[int, int] | list[int]
) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"expected CHW image, got shape {tuple(image.shape)}")
    new_h, new_w = int(size[0]), int(size[1])
    device = image.device
    pil_image = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
    pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)
    return torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).to(device)
