"""
visualizer.py — Match visualization for wafer blob comparison.

Produces a 2-panel PNG:
  Panel 1 — DISK + LightGlue: all LG matches (gray) + RANSAC inliers (green)
  Panel 2 — Verdict bar
"""

import cv2
import numpy as np

GAP      = 20      # pixels between the two images in each panel
MARGIN   = 12      # outer margin
FONT     = cv2.FONT_HERSHEY_SIMPLEX

DISK_THRESHOLD = 15


# ── low-level helpers ─────────────────────────────────────────────────────────

def _side_by_side(rgb0: np.ndarray, rgb1: np.ndarray,
                  bg: tuple = (245, 245, 245)) -> tuple[np.ndarray, int]:
    """
    Stitch two RGB images side-by-side with a GAP.
    Both images are letterboxed to the same height.
    Returns (canvas_bgr, x_start_of_img1).
    """
    H = max(rgb0.shape[0], rgb1.shape[0])
    W = rgb0.shape[1] + GAP + rgb1.shape[1]
    canvas = np.full((H, W, 3), bg, dtype=np.uint8)

    bgr0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR)
    canvas[:rgb0.shape[0], :rgb0.shape[1]] = bgr0

    x1 = rgb0.shape[1] + GAP
    bgr1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR)
    canvas[:rgb1.shape[0], x1:x1 + rgb1.shape[1]] = bgr1

    return canvas, x1


def _draw_lines_alpha(canvas: np.ndarray, pts0, pts1, color, alpha: float,
                      thickness: int = 1) -> None:
    """Draw a batch of lines with alpha blending onto canvas (in-place)."""
    if len(pts0) == 0:
        return
    overlay = canvas.copy()
    for p0, p1 in zip(pts0, pts1):
        cv2.line(overlay,
                 (int(p0[0]), int(p0[1])),
                 (int(p1[0]), int(p1[1])),
                 color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)


def _draw_circles(canvas: np.ndarray, pts, radius: int = 4,
                  color=(0, 200, 0), thickness: int = 1) -> None:
    for p in pts:
        cv2.circle(canvas, (int(p[0]), int(p[1])), radius, color, thickness, cv2.LINE_AA)


def _draw_inlier_lines(canvas: np.ndarray, pts0, pts1, x1_offset: int,
                       color=(0, 210, 60), thickness: int = 2) -> None:
    for p0, p1 in zip(pts0, pts1):
        a = (int(p0[0]),              int(p0[1]))
        b = (int(p1[0]) + x1_offset, int(p1[1]))
        cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, a, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, b, 4, color, -1, cv2.LINE_AA)


def _header_bar(width: int, text: str, n_matches: int, n_inliers: int,
                bg_color=(40, 40, 40)) -> np.ndarray:
    bar = np.full((36, width, 3), bg_color, dtype=np.uint8)
    match_txt = f"  {text}  |  all matches: {n_matches}  |  RANSAC inliers: {n_inliers}"
    cv2.putText(bar, match_txt, (8, 24), FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return bar


def _verdict_bar(width: int, n_disk: int,
                 path0: str = "", path1: str = "",
                 query_rotation: float = 0.0) -> np.ndarray:
    import os
    name0 = os.path.basename(path0)
    name1 = os.path.basename(path1)

    same = n_disk >= DISK_THRESHOLD

    if same:
        bg   = (20, 120, 20)
        icon = "✓  SAME BLOB"
    else:
        bg   = (100, 30, 20)
        icon = "✗  DIFFERENT  (or add more reference images)"

    rot_suffix = f"   rotation: {query_rotation:+.0f}°" if query_rotation != 0 else ""
    disk_sym   = "✓" if same else "✗"

    bar = np.full((66, width, 3), bg, dtype=np.uint8)
    cv2.putText(bar, f"  {icon}{rot_suffix}",
                (8, 22), FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar,
                f"  DISK+LG: {n_disk} inliers {disk_sym} (thr={DISK_THRESHOLD})",
                (8, 44), FONT, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(bar, f"  {name0}   vs   {name1}",
                (8, 60), FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    return bar


# ── RANSAC helper ─────────────────────────────────────────────────────────────

def _ransac_split(kps0: np.ndarray, kps1: np.ndarray,
                  thresh: float = 3.0) -> tuple[np.ndarray, np.ndarray, int]:
    if len(kps0) < 4:
        return kps0, kps1, len(kps0)
    _, mask = cv2.findHomography(
        kps0.reshape(-1, 1, 2).astype(np.float32),
        kps1.reshape(-1, 1, 2).astype(np.float32),
        cv2.RANSAC, thresh,
    )
    if mask is None:
        return np.empty((0, 2)), np.empty((0, 2)), 0
    m = mask.ravel().astype(bool)
    return kps0[m], kps1[m], int(m.sum())


# ── DISK+LG panel ─────────────────────────────────────────────────────────────

def _panel_disk(rgb0: np.ndarray, rgb1: np.ndarray,
                feats0: dict, feats1: dict,
                disk_matcher) -> tuple[np.ndarray, int]:
    import torch

    def _t(arr):
        return torch.from_numpy(arr).to(disk_matcher.device).unsqueeze(0)
    def _hw(hw):
        return torch.tensor([[hw[1], hw[0]]], dtype=torch.float32,
                             device=disk_matcher.device)

    data = {
        "image0": {"keypoints": _t(feats0["keypoints"]),
                   "descriptors": _t(feats0["descriptors"]),
                   "image_size": _hw(feats0["hw"])},
        "image1": {"keypoints": _t(feats1["keypoints"]),
                   "descriptors": _t(feats1["descriptors"]),
                   "image_size": _hw(feats1["hw"])},
    }
    with torch.no_grad():
        out = disk_matcher.matcher(data)

    matches = out["matches"][0].cpu().numpy()
    all_kps0 = feats0["keypoints"][matches[:, 0]] if len(matches) else np.empty((0, 2))
    all_kps1 = feats1["keypoints"][matches[:, 1]] if len(matches) else np.empty((0, 2))

    in_kps0, in_kps1, n_in = _ransac_split(all_kps0, all_kps1)

    canvas, x1 = _side_by_side(rgb0, rgb1)

    MAX_DISPLAY = 400
    rng = np.random.default_rng(0)
    def _subsample(a, b, n):
        if len(a) <= n:
            return a, b
        idx = rng.choice(len(a), n, replace=False)
        return a[idx], b[idx]

    notin = np.ones(len(all_kps0), dtype=bool)
    for ik0 in in_kps0:
        hit = np.where((all_kps0 == ik0).all(axis=1))[0]
        if len(hit):
            notin[hit[0]] = False
    non_kps0 = all_kps0[notin]
    non_kps1 = all_kps1[notin]

    non_kps1_shifted = non_kps1.copy(); non_kps1_shifted[:, 0] += x1
    in_kps1_shifted  = in_kps1.copy();  in_kps1_shifted[:, 0] += x1

    d_non0, d_non1s = _subsample(non_kps0, non_kps1_shifted, MAX_DISPLAY // 2)
    _draw_lines_alpha(canvas, d_non0, d_non1s,
                      color=(160, 160, 160), alpha=0.35, thickness=1)
    d_in0, d_in1 = _subsample(in_kps0, in_kps1, MAX_DISPLAY)
    _draw_inlier_lines(canvas, d_in0, d_in1, x1)

    _draw_circles(canvas, feats0["keypoints"], radius=2,
                  color=(200, 200, 200), thickness=-1)
    kps1_shifted = feats1["keypoints"].copy()
    kps1_shifted[:, 0] += x1
    _draw_circles(canvas, kps1_shifted, radius=2,
                  color=(200, 200, 200), thickness=-1)

    H, W = canvas.shape[:2]
    for x_pos, kp_n in [(8, len(feats0["keypoints"])),
                        (x1 + 8, len(feats1["keypoints"]))]:
        cv2.putText(canvas, f"kp: {kp_n}",
                    (x_pos, H - 8), FONT, 0.42, (60, 60, 60), 1, cv2.LINE_AA)

    hdr = _header_bar(W, "DISK + LightGlue", len(matches), n_in)
    return np.vstack([hdr, canvas]), n_in


# ── main API ──────────────────────────────────────────────────────────────────

def build_comparison(
    rgb0: np.ndarray, rgb1: np.ndarray,
    feats0: dict, feats1: dict,
    disk_matcher,
    path0: str = "", path1: str = "",
    query_rotation: float = 0.0,
) -> np.ndarray:
    """
    Build the full comparison image (DISK panel + verdict bar).
    rgb1/feats1 should already be pre-rotated to the optimal orientation.
    Returns a BGR numpy array ready for cv2.imwrite.
    """
    panel_disk, n_disk = _panel_disk(rgb0, rgb1, feats0, feats1, disk_matcher)

    verdict = _verdict_bar(panel_disk.shape[1], n_disk, path0, path1, query_rotation)
    sep     = np.full((3, panel_disk.shape[1], 3), (180, 180, 180), dtype=np.uint8)

    return np.vstack([panel_disk, sep, verdict])
