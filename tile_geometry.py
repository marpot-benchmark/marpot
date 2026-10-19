from __future__ import annotations

from typing import Sequence


def sliding_tiles(
    height: int, width: int, tile_size: Sequence[int], overlap: float
) -> list[tuple[int, int, int, int]]:
    th, tw = int(tile_size[0]), int(tile_size[1])
    if th <= 0 or tw <= 0 or not 0 <= overlap < 1:
        raise ValueError("tile size must be positive and overlap must be in [0, 1)")

    def starts(length: int, size: int) -> list[int]:
        if length <= size:
            return [0]
        stride = max(1, int(round(size * (1.0 - overlap))))
        values = list(range(0, length - size + 1, stride))
        values.append(length - size)
        return sorted(set(values))

    return [
        (y, x, min(y + th, height), min(x + tw, width))
        for y in starts(height, th)
        for x in starts(width, tw)
    ]
