"""
DISK + LightGlue sparse matcher (kornia).

DISK replaces SuperPoint (not available in kornia 0.8) and was specifically
designed to work with LightGlue.  Descriptors are 128-dim.

Features ARE pre-extractable, so query features are extracted once and
reused across all DB comparisons at query time.
"""

import cv2
import numpy as np
import torch
import kornia.feature as KF

RANSAC_THRESH = 3.0   # px


class DISKLightGlueMatcher:
    name = "disk_lg"

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.extractor = KF.DISK.from_pretrained("depth").to(self.device).eval()
        self.matcher   = KF.LightGlue(features="disk").to(self.device).eval()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_tensor(self, gray: np.ndarray) -> torch.Tensor:
        """(H, W) uint8 → [1, 3, H, W] float32 in [0, 1] (DISK expects RGB)."""
        g = gray.astype(np.float32) / 255.0
        rgb3 = np.stack([g, g, g], axis=0)          # (3, H, W)
        return torch.from_numpy(rgb3).unsqueeze(0).to(self.device)  # [1, 3, H, W]

    # ── public API ────────────────────────────────────────────────────────────

    def extract(self, gray: np.ndarray) -> dict:
        """
        Detect and describe keypoints with DISK.
        Returns serializable dict with keypoints, descriptors, scores, hw.
        """
        with torch.no_grad():
            feats_list = self.extractor(
                self._to_tensor(gray), pad_if_not_divisible=True
            )
        f = feats_list[0]   # DISKFeatures for batch item 0
        return {
            "keypoints":   f.keypoints.cpu().numpy().astype(np.float32),    # [N, 2]
            "descriptors": f.descriptors.cpu().numpy().astype(np.float32),  # [N, 128]
            "scores":      f.detection_scores.cpu().numpy().astype(np.float32),  # [N]
            "hw": (int(gray.shape[0]), int(gray.shape[1])),  # (H, W)
        }

    def match(self, feats0: dict, feats1: dict) -> int:
        """
        Match two feature dicts with LightGlue.
        Returns the RANSAC inlier count.
        """
        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr).to(self.device).unsqueeze(0)  # [1, N, D]

        def _hw_tensor(hw: tuple) -> torch.Tensor:
            # LightGlue expects image_size as [W, H] (x, y order)
            return torch.tensor(
                [[hw[1], hw[0]]], dtype=torch.float32, device=self.device
            )

        data = {
            "image0": {
                "keypoints":   _t(feats0["keypoints"]),
                "descriptors": _t(feats0["descriptors"]),
                "image_size":  _hw_tensor(feats0["hw"]),
            },
            "image1": {
                "keypoints":   _t(feats1["keypoints"]),
                "descriptors": _t(feats1["descriptors"]),
                "image_size":  _hw_tensor(feats1["hw"]),
            },
        }

        with torch.no_grad():
            out = self.matcher(data)

        # 'matches' is a list of [M, 2] tensors, one per batch item
        matches = out["matches"][0].cpu().numpy()   # [M, 2]
        if len(matches) < 4:
            return len(matches)

        kps0 = feats0["keypoints"][matches[:, 0]]
        kps1 = feats1["keypoints"][matches[:, 1]]

        _, mask = cv2.findHomography(
            kps0.reshape(-1, 1, 2),
            kps1.reshape(-1, 1, 2),
            cv2.RANSAC, RANSAC_THRESH,
        )
        return int(mask.sum()) if mask is not None else 0

    # ── DB serialization ──────────────────────────────────────────────────────

    def save_features(self, path: str, feats: dict) -> None:
        np.savez_compressed(
            path,
            keypoints=feats["keypoints"],
            descriptors=feats["descriptors"],
            scores=feats["scores"],
            hw=np.array(feats["hw"], dtype=np.int32),
        )

    def load_features(self, path: str) -> dict:
        d = np.load(path)
        return {
            "keypoints":   d["keypoints"],
            "descriptors": d["descriptors"],
            "scores":      d["scores"],
            "hw":          tuple(d["hw"].tolist()),
        }
