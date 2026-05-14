"""
LoFTR dense matcher (kornia).

LoFTR performs joint feature extraction + matching; features cannot be
pre-extracted per image.  The DB therefore stores the preprocessed
grayscale image as a .npy file and reruns inference at query time.

Pretrained weights: 'indoor' (default) or 'outdoor'.
"""

import cv2
import numpy as np
import torch
import kornia.feature as KF

RANSAC_THRESH  = 3.0   # px, for homography RANSAC
MIN_CONF       = 0.5   # LoFTR confidence threshold
LOFTR_MAX_SIDE = 640   # internal resize to keep inference fast on CPU


class LoFTRMatcher:
    name = "loftr"

    def __init__(self, pretrained: str = "indoor", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = KF.LoFTR(pretrained=pretrained).to(self.device).eval()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resize(self, gray: np.ndarray) -> np.ndarray:
        H, W = gray.shape
        if max(H, W) <= LOFTR_MAX_SIDE:
            return gray
        s = LOFTR_MAX_SIDE / max(H, W)
        return cv2.resize(gray, (int(W * s), int(H * s)),
                          interpolation=cv2.INTER_AREA)

    def _to_tensor(self, gray: np.ndarray) -> torch.Tensor:
        """(H, W) uint8 → [1, 1, H, W] float32 in [0, 1]."""
        g = self._resize(gray)
        t = torch.from_numpy(g.astype(np.float32) / 255.0)
        return t.unsqueeze(0).unsqueeze(0).to(self.device)

    # ── public API ────────────────────────────────────────────────────────────

    def match(self, gray0: np.ndarray, gray1: np.ndarray) -> int:
        """
        Run LoFTR on a pair of grayscale images.
        Returns the RANSAC inlier count.
        """
        with torch.no_grad():
            out = self.model({
                "image0": self._to_tensor(gray0),
                "image1": self._to_tensor(gray1),
            })

        kps0 = out["keypoints0"].cpu().numpy()   # [N, 2]
        kps1 = out["keypoints1"].cpu().numpy()   # [N, 2]
        conf = out["confidence"].cpu().numpy()   # [N]

        keep = conf > MIN_CONF
        kps0, kps1 = kps0[keep], kps1[keep]

        if len(kps0) < 4:
            return len(kps0)

        _, mask = cv2.findHomography(
            kps0.reshape(-1, 1, 2).astype(np.float32),
            kps1.reshape(-1, 1, 2).astype(np.float32),
            cv2.RANSAC, RANSAC_THRESH,
        )
        return int(mask.sum()) if mask is not None else 0

    # ── DB serialization ──────────────────────────────────────────────────────

    def extract(self, gray: np.ndarray) -> dict:
        """LoFTR has no standalone extraction step — store the image itself."""
        return {"gray": gray}

    def save_features(self, path: str, feats: dict) -> None:
        np.save(path, feats["gray"])

    def load_features(self, path: str) -> dict:
        return {"gray": np.load(path)}
