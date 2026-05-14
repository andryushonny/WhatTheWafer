"""
preprocessing.py — Wafer blob image preprocessing.

Pipeline:
  1. Load image → RGB uint8  (TIF, JPEG, PNG, BMP)
  2. Resize longest side to 1024 px (INTER_AREA, never upscale)
  3. Segment blob: HSV saturation + LAB chromatic distance (Otsu union)
  4. Morphological cleanup
  5. Crop to bounding box + 10 % padding
  6. Return rgb_crop + gray_crop (grayscale for matcher input)
"""

import logging
import cv2
import numpy as np
import tifffile

logging.getLogger("tifffile").setLevel(logging.ERROR)

MAX_SIDE = 1024
PAD_FRAC = 0.10


# ── Loading ───────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """
    Load TIF / JPEG / PNG / BMP → RGB uint8.
    Tries tifffile first (preserves 16-bit / OME metadata for TIF).
    Falls back to cv2 for JPEG, PNG and other formats.
    """
    img = None
    try:
        img = tifffile.imread(path)
    except Exception:
        pass

    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        # cv2 loads as BGR (or BGRA) — convert to RGB
        if img.ndim == 3 and img.shape[2] in (3, 4):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if img.shape[2] == 3
                                    else cv2.COLOR_BGRA2RGB)

    # Normalise to (H, W, 3) uint8
    while img.ndim > 3:
        img = img[0]
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
        img = np.transpose(img, (1, 2, 0))
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        f = img.astype(np.float32) - img.min()
        mx = f.max()
        img = (f / mx * 255).astype(np.uint8) if mx > 0 else f.astype(np.uint8)
    return img  # RGB uint8


# Keep the old name as an alias so external callers aren't broken
load_tif = load_image


# ── Resize ────────────────────────────────────────────────────────────────────

def resize_max(rgb: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    H, W = rgb.shape[:2]
    if max(H, W) <= max_side:
        return rgb
    s = max_side / max(H, W)
    return cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)


# ── Segmentation ──────────────────────────────────────────────────────────────

def segment_blob(rgb: np.ndarray) -> np.ndarray:
    """
    Isolate the colored blob from the gray silicon substrate.
    Combines HSV saturation threshold and LAB chromatic-distance threshold
    (Otsu on each, then union) to handle both bright and dark-field images.
    Returns uint8 binary mask (255 = blob).
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # HSV saturation
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, mask_sat = cv2.threshold(hsv[:, :, 1], 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # LAB chromatic distance from the gray axis
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = np.sqrt((lab[:, :, 1] - 128) ** 2 + (lab[:, :, 2] - 128) ** 2)
    ab_u8 = (ab / (ab.max() + 1e-6) * 255).astype(np.uint8)
    _, mask_lab = cv2.threshold(ab_u8, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.bitwise_or(mask_sat, mask_lab)

    # Morphological cleanup: close gaps, remove noise
    H, W = rgb.shape[:2]
    k = max(5, min(H, W) // 60) | 1   # odd kernel size, at least 5
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 4, k * 4)),
        iterations=2,
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        iterations=1,
    )
    return mask


# ── Crop ──────────────────────────────────────────────────────────────────────

def crop_blob(
    rgb: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop to blob bounding box + PAD_FRAC padding.
    Falls back to full image if no blob pixels found.
    Returns (rgb_crop, mask_crop, gray_crop).
    """
    H, W = rgb.shape[:2]
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        return rgb.copy(), mask.copy(), gray

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    px = max(1, int(bw * PAD_FRAC))
    py = max(1, int(bh * PAD_FRAC))

    x0 = max(0, x0 - px);  x1 = min(W, x1 + px)
    y0 = max(0, y0 - py);  y1 = min(H, y1 + py)

    rgb_c  = rgb[y0:y1, x0:x1].copy()
    mask_c = mask[y0:y1, x0:x1].copy()
    gray_c = cv2.cvtColor(cv2.cvtColor(rgb_c, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    return rgb_c, mask_c, gray_c


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(path: str, max_side: int = MAX_SIDE) -> dict:
    """
    Full preprocessing pipeline for one image.

    Returns dict:
      rgb    (H, W, 3) uint8 — color crop (keep for display)
      gray   (H, W)   uint8 — grayscale crop (matcher input)
      mask   (H, W)   uint8 — blob mask
      source str      — original path
    """
    rgb = load_image(str(path))
    rgb = resize_max(rgb, max_side)
    mask = segment_blob(rgb)
    rgb_c, mask_c, gray_c = crop_blob(rgb, mask)
    return {"rgb": rgb_c, "gray": gray_c, "mask": mask_c, "source": str(path)}
