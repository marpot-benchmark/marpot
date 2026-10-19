
from __future__ import annotations

import numpy as np


def _xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = b
    return np.array(
        [(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)],
        dtype=np.float32,
    )


def _cxcywh_to_xyxy(z: np.ndarray) -> np.ndarray:
    cx, cy, w, h = z[:4]
    w = max(float(w), 1e-3)
    h = max(float(h), 1e-3)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


class KalmanBoxTracker:
    def __init__(self, bbox: np.ndarray, nsa: bool = True):
        z = _xyxy_to_cxcywh(bbox)
        self.mean = np.concatenate([z, np.zeros(4, dtype=np.float32)])
        self.cov = np.eye(8, dtype=np.float32)
        self.cov[4:, 4:] *= 1000.0
        self.cov *= 10.0
        self.nsa = nsa

        self._q_pos = 1.0
        self._q_vel = 0.01
        self._r_pos = 1.0

    def predict_bbox(self, steps: float) -> np.ndarray:
        s = float(steps)
        cx = self.mean[0] + self.mean[4] * s
        cy = self.mean[1] + self.mean[5] * s
        w = self.mean[2] + self.mean[6] * s
        h = self.mean[3] + self.mean[7] * s
        return _cxcywh_to_xyxy(np.array([cx, cy, w, h], dtype=np.float32))

    def reset_velocity(self):
        self.mean[4:] = 0.0
        self.cov[4:, 4:] = np.eye(4, dtype=np.float32) * 1000.0

    def update(self, bbox: np.ndarray, score: float = 1.0, steps: float = 1.0):
        s = float(max(steps, 1.0))

        F = np.eye(8, dtype=np.float32)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = s

        Q = np.zeros((8, 8), dtype=np.float32)
        np.fill_diagonal(Q[:4, :4], self._q_pos * s)
        np.fill_diagonal(Q[4:, 4:], self._q_vel * s)

        self.mean = F @ self.mean
        self.cov = F @ self.cov @ F.T + Q

        H = np.zeros((4, 8), dtype=np.float32)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0

        r_scale = (1.0 - min(max(score, 0.0), 0.99)) if self.nsa else 1.0
        R = np.eye(4, dtype=np.float32) * self._r_pos * max(r_scale, 1e-2)

        z = _xyxy_to_cxcywh(bbox)
        y = z - H @ self.mean
        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)
        self.mean = self.mean + K @ y
        self.cov = (np.eye(8, dtype=np.float32) - K @ H) @ self.cov

    @property
    def bbox(self) -> np.ndarray:
        return _cxcywh_to_xyxy(self.mean[:4])
