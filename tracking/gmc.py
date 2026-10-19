
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


class GMC:
    def __init__(self, method: str = "sparseOptFlow", downscale: int = 2):
        self.method = method
        self.downscale = max(1, int(downscale))
        self._prev_gray: Optional[np.ndarray] = None
        self.available = cv2 is not None

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        if self.downscale > 1:
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))
        return gray

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.available:
            return IDENTITY_AFFINE.copy()

        gray = self._to_gray(frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            return IDENTITY_AFFINE.copy()

        affine = IDENTITY_AFFINE.copy()
        try:
            if self.method == "ecc":
                affine = self._ecc(self._prev_gray, gray)
            elif self.method == "orb_ransac":
                affine = self._orb(self._prev_gray, gray)
            else:
                affine = self._sparse_optflow(self._prev_gray, gray)
        except Exception:
            affine = IDENTITY_AFFINE.copy()

        self._prev_gray = gray
        return affine

    def _rescale(self, affine: np.ndarray) -> np.ndarray:
        if self.downscale > 1:
            affine = affine.copy()
            affine[0, 2] *= self.downscale
            affine[1, 2] *= self.downscale
        return affine

    def _sparse_optflow(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(
            prev, maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3
        )
        if pts is None or len(pts) < 4:
            return IDENTITY_AFFINE.copy()
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None)
        good = status.ravel().astype(bool)
        p0, p1 = pts[good], nxt[good]
        if len(p0) < 4:
            return IDENTITY_AFFINE.copy()
        affine, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC)
        if affine is None:
            return IDENTITY_AFFINE.copy()
        return self._rescale(affine.astype(np.float32))

    def _orb(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        orb = cv2.ORB_create(nfeatures=1000)
        k1, d1 = orb.detectAndCompute(prev, None)
        k2, d2 = orb.detectAndCompute(cur, None)
        if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
            return IDENTITY_AFFINE.copy()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        if len(matches) < 4:
            return IDENTITY_AFFINE.copy()
        p0 = np.float32([k1[m.queryIdx].pt for m in matches])
        p1 = np.float32([k2[m.trainIdx].pt for m in matches])
        affine, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC)
        if affine is None:
            return IDENTITY_AFFINE.copy()
        return self._rescale(affine.astype(np.float32))

    def _ecc(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            50,
            1e-4,
        )
        try:
            _, warp = cv2.findTransformECC(
                prev, cur, warp, cv2.MOTION_AFFINE, criteria, None, 1
            )
        except cv2.error:
            return IDENTITY_AFFINE.copy()
        return self._rescale(warp.astype(np.float32))


def warp_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    homog = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    return (homog @ affine.T).reshape(points.shape)


def warp_bbox(bbox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
    )
    w = warp_points(corners, affine)
    return np.array(
        [w[:, 0].min(), w[:, 1].min(), w[:, 0].max(), w[:, 1].max()],
        dtype=np.float32,
    )
