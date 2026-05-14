"""
visualizer.py — Match visualization for wafer blob comparison.

Produces a 3-panel PNG:
  Panel 1 — DISK + LightGlue: all LG matches (gray) + RANSAC inliers (green)
  Panel 2 — LoFTR: all conf>0.5 matches (rainbow by x-position) + RANSAC inliers (green)
  Panel 3 — Verdict bar
"""

import cv2
import numpy as np

GAP      = 20      # pixels between the two images in each panel
MARGIN   = 12      # outer margin
FONT     = cv2.FONT_HERSHEY_SIMPLEX

# DISK+LG cross-blob noise floor: 0–4 → threshold 15 is conservative
DISK_THRESHOLD  = 15
# LoFTR cross-blob noise floor: 10–40 → need a much higher bar
LOFTR_THRESHOLD = 100


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

    # Place img0 (top-aligned)
    bgr0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR)
    canvas[:rgb0.shape[0], :rgb0.shape[1]] = bgr0

    # Place img1
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


def _draw_lines_rainbow(canvas: np.ndarray, pts0, pts1, x1: int,
                        img_width: int, alpha: float = 0.55) -> None:
    """
    Draw lines coloured by the horizontal position of their left endpoint.
    Gives a characteristic 'spectral flow' look for LoFTR matches.
    """
    if len(pts0) == 0:
        return
    overlay = canvas.copy()
    cmap = _build_rainbow_lut()
    for p0, p1 in zip(pts0, pts1):
        t = np.clip(p0[0] / max(img_width - 1, 1), 0, 1)
        idx = int(t * 255)
        color = (int(cmap[idx, 0]),
                 int(cmap[idx, 1]),
                 int(cmap[idx, 2]))
        cv2.line(overlay,
                 (int(p0[0]), int(p0[1])),
                 (int(p1[0]) + x1, int(p1[1])),
                 color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)


def _build_rainbow_lut() -> np.ndarray:
    """256-entry BGR lookup table cycling blue→cyan→green→yellow→red."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        h = int(240 - i * 240 / 255)   # hue: 240 (blue) → 0 (red)
        hsv = np.uint8([[[h, 230, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        lut[i] = bgr
    return lut


def _draw_circles(canvas: np.ndarray, pts, radius: int = 4,
                  color=(0, 200, 0), thickness: int = 1) -> None:
    for p in pts:
        cv2.circle(canvas, (int(p[0]), int(p[1])), radius, color, thickness, cv2.LINE_AA)


def _draw_inlier_lines(canvas: np.ndarray, pts0, pts1, x1_offset: int,
                       color=(0, 210, 60), thickness: int = 2) -> None:
    for p0, p1 in zip(pts0, pts1):
        a = (int(p0[0]),          int(p0[1]))
        b = (int(p1[0]) + x1_offset, int(p1[1]))
        cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, a, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, b, 4, color, -1, cv2.LINE_AA)


def _header_bar(width: int, text: str, n_matches: int, n_inliers: int,
                bg_color=(40, 40, 40)) -> np.ndarray:
    """Create a dark header bar with text."""
    bar = np.full((36, width, 3), bg_color, dtype=np.uint8)
    match_txt = f"  {text}  |  all matches: {n_matches}  |  RANSAC inliers: {n_inliers}"
    cv2.putText(bar, match_txt, (8, 24), FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return bar


def _verdict_bar(width: int, n_disk: int, n_loftr: int,
                 path0: str = "", path1: str = "",
                 query_rotation: float = 0.0) -> np.ndarray:
    """
    Verdict bar with per-matcher decisions.
    DISK+LG is the primary (more discriminative) indicator.
    LoFTR is secondary (higher noise floor, needs a stricter threshold).
    """
    import os
    name0 = os.path.basename(path0)
    name1 = os.path.basename(path1)

    disk_match  = n_disk  >= DISK_THRESHOLD
    loftr_match = n_loftr >= LOFTR_THRESHOLD
    same        = disk_match or loftr_match

    if same:
        bg   = (20, 120, 20)
        icon = "✓  SAME BLOB"
    else:
        bg   = (100, 30, 20)
        icon = "✗  DIFFERENT  (or add more reference images)"

    rot_suffix = f"   rotation: {query_rotation:+.0f}°" if query_rotation != 0 else ""
    disk_sym  = "✓" if disk_match  else "✗"
    loftr_sym = "✓" if loftr_match else "✗"

    bar = np.full((66, width, 3), bg, dtype=np.uint8)
    cv2.putText(bar, f"  {icon}{rot_suffix}",
                (8, 22), FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar,
                f"  DISK+LG: {n_disk} inliers {disk_sym} (thr={DISK_THRESHOLD})   "
                f"LoFTR: {n_loftr} inliers {loftr_sym} (thr={LOFTR_THRESHOLD})",
                (8, 44), FONT, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(bar, f"  {name0}   vs   {name1}",
                (8, 60), FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    return bar


# ── RANSAC helper ─────────────────────────────────────────────────────────────

def _ransac_split(kps0: np.ndarray, kps1: np.ndarray,
                  thresh: float = 3.0) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Run RANSAC homography on raw match arrays.
    Returns (inlier_kps0, inlier_kps1, inlier_count).
    """
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


# ── per-matcher panels ────────────────────────────────────────────────────────

def _panel_disk(rgb0: np.ndarray, rgb1: np.ndarray,
                feats0: dict, feats1: dict,
                disk_matcher) -> tuple[np.ndarray, int]:
    """
    Draw DISK + LightGlue panel.
    Returns (panel_bgr, n_inliers).
    """
    import torch

    # Run LightGlue
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

    matches = out["matches"][0].cpu().numpy()   # [M, 2]
    all_kps0 = feats0["keypoints"][matches[:, 0]] if len(matches) else np.empty((0, 2))
    all_kps1 = feats1["keypoints"][matches[:, 1]] if len(matches) else np.empty((0, 2))

    in_kps0, in_kps1, n_in = _ransac_split(all_kps0, all_kps1)

    # Canvas
    canvas, x1 = _side_by_side(rgb0, rgb1)

    # Subsample displayed lines to keep the image readable
    MAX_DISPLAY = 400
    rng = np.random.default_rng(0)
    def _subsample(a, b, n):
        if len(a) <= n:
            return a, b
        idx = rng.choice(len(a), n, replace=False)
        return a[idx], b[idx]

    # Draw all LG matches (non-inliers) in semi-transparent gray
    notin = np.ones(len(all_kps0), dtype=bool)
    # Mark inlier indices — compare positions
    for ik0 in in_kps0:
        hit = np.where((all_kps0 == ik0).all(axis=1))[0]
        if len(hit):
            notin[hit[0]] = False
    non_kps0 = all_kps0[notin]
    non_kps1 = all_kps1[notin]

    # Shift img1 keypoints by x1
    non_kps1_shifted = non_kps1.copy(); non_kps1_shifted[:, 0] += x1
    in_kps1_shifted  = in_kps1.copy();  in_kps1_shifted[:, 0] += x1

    d_non0, d_non1s = _subsample(non_kps0, non_kps1_shifted, MAX_DISPLAY // 2)
    _draw_lines_alpha(canvas, d_non0, d_non1s,
                      color=(160, 160, 160), alpha=0.35, thickness=1)
    d_in0, d_in1 = _subsample(in_kps0, in_kps1, MAX_DISPLAY)
    _draw_inlier_lines(canvas, d_in0, d_in1, x1)

    # Light keypoint dots for all detected points
    _draw_circles(canvas, feats0["keypoints"], radius=2,
                  color=(200, 200, 200), thickness=-1)
    kps1_shifted = feats1["keypoints"].copy()
    kps1_shifted[:, 0] += x1
    _draw_circles(canvas, kps1_shifted, radius=2,
                  color=(200, 200, 200), thickness=-1)

    # Stats text on canvas
    H, W = canvas.shape[:2]
    for x_pos, kp_n in [(8, len(feats0["keypoints"])),
                        (x1 + 8, len(feats1["keypoints"]))]:
        cv2.putText(canvas, f"kp: {kp_n}",
                    (x_pos, H - 8), FONT, 0.42, (60, 60, 60), 1, cv2.LINE_AA)

    hdr = _header_bar(W, "DISK + LightGlue", len(matches), n_in)
    return np.vstack([hdr, canvas]), n_in


def _panel_loftr(rgb0: np.ndarray, rgb1: np.ndarray,
                 gray0: np.ndarray, gray1: np.ndarray,
                 loftr_matcher) -> tuple[np.ndarray, int]:
    """
    Draw LoFTR panel with rainbow-coloured matches.
    Returns (panel_bgr, n_inliers).
    """
    import torch

    MIN_CONF = 0.5
    t0 = loftr_matcher._to_tensor(gray0)
    t1 = loftr_matcher._to_tensor(gray1)

    with torch.no_grad():
        out = loftr_matcher.model({"image0": t0, "image1": t1})

    kps0 = out["keypoints0"].cpu().numpy()
    kps1 = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()

    # Scale keypoints back to original gray dimensions
    # (LoFTR internally works on a potentially resized image)
    from matchers.loftr_matcher import LOFTR_MAX_SIDE
    import math
    H0, W0 = gray0.shape
    H1, W1 = gray1.shape
    lmax = LOFTR_MAX_SIDE
    s0 = lmax / max(H0, W0) if max(H0, W0) > lmax else 1.0
    s1 = lmax / max(H1, W1) if max(H1, W1) > lmax else 1.0
    kps0 = kps0 / s0
    kps1 = kps1 / s1

    keep = conf > MIN_CONF
    kps0_f, kps1_f = kps0[keep], kps1[keep]

    in_kps0, in_kps1, n_in = _ransac_split(kps0_f, kps1_f)

    canvas, x1 = _side_by_side(rgb0, rgb1)

    # Rainbow: sample of conf-filtered matches (background context)
    MAX_DISP = 600
    rng = np.random.default_rng(0)
    def _sub(a, b, n):
        if len(a) <= n:
            return a, b
        idx = rng.choice(len(a), n, replace=False)
        return a[idx], b[idx]

    d_kps0_f, d_kps1_f = _sub(kps0_f, kps1_f, MAX_DISP)
    _draw_lines_rainbow(canvas, d_kps0_f, d_kps1_f, x1,
                        img_width=rgb0.shape[1], alpha=0.5)

    # Green: RANSAC inliers on top (also capped for readability)
    d_in0, d_in1 = _sub(in_kps0, in_kps1, MAX_DISP)
    _draw_inlier_lines(canvas, d_in0, d_in1, x1)

    H, W = canvas.shape[:2]
    cv2.putText(canvas,
                f"conf>{MIN_CONF:.1f}: {keep.sum()}  |  inliers: {n_in}",
                (8, H - 8), FONT, 0.42, (60, 60, 60), 1, cv2.LINE_AA)

    hdr = _header_bar(W, "LoFTR (dense)", int(keep.sum()), n_in)
    return np.vstack([hdr, canvas]), n_in


# ── main API ──────────────────────────────────────────────────────────────────

def build_comparison(
    rgb0: np.ndarray, rgb1: np.ndarray,
    gray0: np.ndarray, gray1: np.ndarray,
    feats0: dict, feats1: dict,
    loftr_matcher, disk_matcher,
    path0: str = "", path1: str = "",
    query_rotation: float = 0.0,
) -> np.ndarray:
    """
    Build the full comparison image (3 panels stacked vertically).
    rgb1/gray1/feats1 should already be pre-rotated to the optimal orientation.
    Returns a BGR numpy array ready for cv2.imwrite.
    """
    panel_disk, n_disk = _panel_disk(rgb0, rgb1, feats0, feats1, disk_matcher)
    panel_loftr, n_loftr = _panel_loftr(rgb0, rgb1, gray0, gray1, loftr_matcher)

    # Unify widths (pad narrower panel to match the wider)
    W = max(panel_disk.shape[1], panel_loftr.shape[1])

    def _pad_width(img, target_w, bg=(245, 245, 245)):
        if img.shape[1] >= target_w:
            return img
        pad = np.full((img.shape[0], target_w - img.shape[1], 3), bg, dtype=np.uint8)
        return np.hstack([img, pad])

    panel_disk  = _pad_width(panel_disk,  W)
    panel_loftr = _pad_width(panel_loftr, W)

    verdict = _verdict_bar(W, n_disk, n_loftr, path0, path1, query_rotation)

    # Thin separator lines
    sep = np.full((3, W, 3), (180, 180, 180), dtype=np.uint8)

    combined = np.vstack([panel_disk, sep, panel_loftr, sep, verdict])
    return combined
