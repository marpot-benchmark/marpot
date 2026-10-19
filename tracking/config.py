
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Dict


@dataclass
class CostWeights:
    iou: float = 0.6
    query: float = 0.3
    center: float = 0.1
    area: float = 0.0
    appearance: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class TrackerConfig:
    type: str = "motion_iou"
    motion_free_ablation: bool = False
    class_agnostic: bool = False

    iou_kind: str = "mask"
    mask_iou_threshold: float = 0.3
    bbox_iou_threshold: float = 0.1
    center_dist_factor: float = 2.0
    area_change_max: float = 4.0
    match_cost_threshold: float = 0.9

    delta_t_cap: int = 10
    scene_cut_gap: int = 50
    max_age_samples: int = 20
    max_age_frames: int = 60
    min_hits: int = 3

    min_score_ship: float = 0.4
    min_score_obstacle: float = 0.3
    ship_class_id: int = 0
    obstacle_class_id: int = 1

    byte_enabled: bool = False
    high_thr: float = 0.5
    low_thr: float = 0.1

    kalman_enabled: bool = False
    nsa_kalman: bool = True
    ocm_enabled: bool = False
    oru_enabled: bool = False
    ocr_enabled: bool = False
    ocm_weight: float = 0.2

    use_query_embed: bool = False
    query_cost_weight: float = 0.5
    query_probe_required: bool = True

    appearance_enabled: bool = False
    appearance_source: str = "mask_pooled"
    appearance_weight: float = 0.3
    appearance_ema: float = 0.9

    sea_prior_enabled: bool = False
    sea_overlap_min: float = 0.1
    sea_margin_frac: float = 0.5
    horizon_band_enabled: bool = False
    gmc_enabled: bool = False
    gmc_method: str = "sparseOptFlow"
    min_track_len: int = 1

    cost_weights: CostWeights = field(default_factory=CostWeights)

    @classmethod
    def from_dict(cls, data: Dict) -> "TrackerConfig":
        if data is None:
            return cls()
        data = dict(data)
        cost = data.pop("cost_weights", None)
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(
                f"Unknown tracker config keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid)}"
            )
        cfg = cls(**data)
        if cost is not None:
            if isinstance(cost, CostWeights):
                cfg.cost_weights = cost
            else:
                cw_valid = {f.name for f in fields(CostWeights)}
                cw_unknown = set(cost) - cw_valid
                if cw_unknown:
                    raise ValueError(f"Unknown cost_weights keys: {sorted(cw_unknown)}")
                cfg.cost_weights = CostWeights(**cost)
        cost_keys = set(cost) if isinstance(cost, dict) else set()
        if "query_cost_weight" in data and "query" not in cost_keys:
            cfg.cost_weights.query = float(data["query_cost_weight"])
        if "appearance_weight" in data and "appearance" not in cost_keys:
            cfg.cost_weights.appearance = float(data["appearance_weight"])
        return cfg

    def as_dict(self) -> Dict:
        d = asdict(self)
        return d

    def min_score_for_class(self, class_id: int) -> float:
        if class_id == self.ship_class_id:
            return self.min_score_ship
        if class_id == self.obstacle_class_id:
            return self.min_score_obstacle
        return min(self.min_score_ship, self.min_score_obstacle)
