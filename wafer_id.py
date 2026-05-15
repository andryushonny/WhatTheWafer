"""
wafer_id.py — Wafer blob fingerprint identification
Using LoFTR (dense) and DISK+LightGlue (sparse) matchers.

Commands:
  add    Add a reference image to the database
  query  Identify an unknown image
  list   List all blobs in the database
  clear  Remove a blob (or the whole DB)

Examples:
  python wafer_id.py add wafers/wafer1_1.tif --name wafer1
  python wafer_id.py add wafers/wafer1_2.tif --name wafer1
  python wafer_id.py add wafers/wafer2_1.tif --name wafer2
  python wafer_id.py query wafers/wafer1_3.tif
  python wafer_id.py query wafers/wafer1_3.tif --matcher loftr
  python wafer_id.py query wafers/wafer1_3.tif --matcher both --debug
  python wafer_id.py list
  python wafer_id.py clear --name wafer1
  python wafer_id.py clear --all
  python wafer_id.py compare wafers/wafer1_1.tif wafers/wafer1_3.tif
  python wafer_id.py compare wafers/wafer1_1.tif wafers/wafer2_1.tif --output out.png
"""

import argparse
import sys
import os

import cv2
import numpy as np
import torch

# Use bundled weights in models/ only if ALL weight files are present.
# If any file is missing, leave torch hub pointing at the default
# ~/.cache/torch/hub/checkpoints/ — weights there will be used, or
# downloaded automatically on first run.
_local_models = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_bundled_weights = [
    "checkpoints/depth-save.pth",
    "checkpoints/disk_lightglue_v0-1_arxiv-pth",
]
if all(os.path.isfile(os.path.join(_local_models, w)) for w in _bundled_weights):
    torch.hub.set_dir(_local_models)

from preprocessing import preprocess
from database import BlobDB
from matchers.disk_lightglue_matcher import DISKLightGlueMatcher

UNKNOWN_THRESHOLD      = 15    # min RANSAC inliers to claim a positive match
DEFAULT_DB             = "database"
ROTATION_ANGLES        = [0, 90, 180, 270]           # coarse cardinal rotations
FINE_ROTATION_OFFSETS  = [-15, -10, -5, 5, 10, 15]  # degrees around best coarse angle
FAST_TOP_K             = 5    # FAISS first-stage candidate count


# ── device + model loading ────────────────────────────────────────────────────

def _device(no_gpu: bool = False) -> str:
    if no_gpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_matcher(device: str, verbose: bool = True) -> DISKLightGlueMatcher:
    if verbose:
        print(f"[*] Loading DISK+LightGlue on {device}...")
    return DISKLightGlueMatcher(device=device)


# ── rotation utilities ────────────────────────────────────────────────────────

def _rotate_gray(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate grayscale image by angle degrees (counter-clockwise).
    Multiples of 90° use lossless np.rot90; arbitrary angles use warpAffine."""
    if angle == 0:
        return gray
    if angle % 90 == 0:
        return np.rot90(gray, int(angle) // 90)
    H, W = gray.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), -angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nW = int(H * sin_a + W * cos_a)
    nH = int(H * cos_a + W * sin_a)
    M[0, 2] += (nW - W) / 2
    M[1, 2] += (nH - H) / 2
    return cv2.warpAffine(gray, M, (nW, nH),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _rotate_rgb(rgb: np.ndarray, angle: float) -> np.ndarray:
    """Rotate RGB image by angle degrees (counter-clockwise)."""
    if angle == 0:
        return rgb
    if angle % 90 == 0:
        return np.rot90(rgb, int(angle) // 90)
    H, W = rgb.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), -angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nW = int(H * sin_a + W * cos_a)
    nH = int(H * cos_a + W * sin_a)
    M[0, 2] += (nW - W) / 2
    M[1, 2] += (nH - H) / 2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(
        cv2.warpAffine(bgr, M, (nW, nH),
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
        cv2.COLOR_BGR2RGB,
    )


# ── matching core ─────────────────────────────────────────────────────────────

def run_matching(
    gray_query: "np.ndarray",
    db: BlobDB,
    disk: DISKLightGlueMatcher,
    verbose: bool = False,
    fine_rotation: bool = False,
) -> list[tuple[str, int, float]]:
    """
    Match gray_query against every reference image in the DB with DISK+LightGlue.

    Tries all 4 cardinal rotations of the query, then optionally searches
    ±5/10/15° around the best cardinal rotation (fine_rotation mode).

    Returns list of (blob_id, score, best_angle_degrees) sorted descending.
    """
    # Pre-extract features for all 4 cardinal rotations of the query
    q_disk_by_angle = {
        angle: disk.extract(_rotate_gray(gray_query, angle))
        for angle in ROTATION_ANGLES
    }

    # FAISS first stage: narrow to a short-list of candidates
    q_descs_by_angle = {a: q["descriptors"] for a, q in q_disk_by_angle.items()}
    candidate_ids = db.fast_candidates(q_descs_by_angle, top_k=FAST_TOP_K)
    if verbose:
        print(f"  [FAISS] candidates: {candidate_ids}")

    all_feats = db.load_blob_features(candidate_ids)
    per_blob: dict[str, tuple[int, float]] = {}

    for blob_id, ref_list in all_feats.items():
        best_score = 0
        best_angle = 0.0
        for i, ref in enumerate(ref_list):
            ref_disk  = 0
            ref_angle = 0.0

            for angle in ROTATION_ANGLES:
                s = disk.match(ref["disk"], q_disk_by_angle[angle])
                if verbose:
                    print(f"    {blob_id}[{i}]  DISK+LG@{angle:3d}°={s}")
                if s > ref_disk:
                    ref_disk  = s
                    ref_angle = float(angle)

            if fine_rotation:
                for offset in FINE_ROTATION_OFFSETS:
                    fine_angle = ref_angle + offset
                    q_fine = disk.extract(_rotate_gray(gray_query, fine_angle))
                    s = disk.match(ref["disk"], q_fine)
                    if verbose:
                        print(f"    {blob_id}[{i}]  DISK+LG@{fine_angle:+.0f}°={s}")
                    if s > ref_disk:
                        ref_disk  = s
                        ref_angle = fine_angle

            if ref_disk > best_score:
                best_score = ref_disk
                best_angle = ref_angle

        per_blob[blob_id] = (best_score, best_angle)

    return sorted(
        [(bid, sc, ang) for bid, (sc, ang) in per_blob.items()],
        key=lambda x: x[1], reverse=True,
    )


def _print_results(
    scores: list[tuple[str, int, float]], top_k: int = 5, threshold: int = UNKNOWN_THRESHOLD
) -> None:
    if not scores:
        print("[?] No results.")
        return

    print()
    print(f"  {'Rank':<5} {'Blob ID':<24} {'Inliers':>8}  {'Rotation':>10}")
    print("  " + "-" * 54)
    for i, (blob_id, score, angle) in enumerate(scores[:top_k]):
        marker = "  ◀" if i == 0 else ""
        angle_str = f"{angle:+.0f}°" if angle != 0 else "0°"
        print(f"  {i+1:<5} {blob_id:<24} {score:>8}  {angle_str:>10}{marker}")
    print()

    best_id, best_score, best_angle = scores[0]
    angle_note = f" (query rotated {best_angle:+.0f}°)" if best_angle != 0 else ""
    if best_score >= threshold:
        print(f"[✓] MATCH → '{best_id}'  ({best_score} inliers{angle_note})")
    else:
        print(
            f"[?] UNKNOWN / NEW BLOB  "
            f"(best: '{best_id}' with {best_score} inliers  <  threshold {threshold})"
        )


# ── add ───────────────────────────────────────────────────────────────────────

def cmd_add(args) -> None:
    if not os.path.isfile(args.image):
        sys.exit(f"[!] File not found: {args.image}")

    db = BlobDB(args.db)

    # Resolve target blob_id
    if args.wafer:
        # --wafer UUID : add another view to an existing wafer
        blob_id = db.resolve_blob_id(args.wafer)
        if blob_id is None:
            sys.exit(f"[!] Wafer '{args.wafer}' not found. Run 'list' to see names and UUIDs.")
        adding_to_existing = True
    else:
        # --new NAME : create a new wafer
        blob_id = args.new
        if db.get_blob(blob_id) is not None:
            existing = db.get_blob(blob_id)
            sys.exit(
                f"[!] Wafer '{blob_id}' already exists "
                f"(uuid:{existing['short_id']}).\n"
                f"    To add another view use:  whattw add {args.image} --wafer {blob_id}"
            )
        adding_to_existing = False

    device = _device(getattr(args, "no_gpu", False))
    disk = _load_matcher(device)

    print(f"[*] Preprocessing {args.image} ...")
    proc = preprocess(args.image)
    rgb, gray = proc["rgb"], proc["gray"]
    print(f"    Cropped shape: {gray.shape[1]}×{gray.shape[0]} px")

    # Duplicate-detection (only meaningful when creating a new wafer)
    if not adding_to_existing and db.blob_ids():
        print(f"[*] Checking against {len(db.blob_ids())} existing blob(s) ...")
        scores = run_matching(gray, db, disk, verbose=args.debug)
        if scores:
            best_id, best_score, _ = scores[0]
            if best_score >= UNKNOWN_THRESHOLD:
                best_blob = db.get_blob(best_id)
                print(
                    f"\n[!] WARNING: image looks like existing wafer '{best_id}' "
                    f"[{best_blob['short_id']}]  ({best_score} inliers ≥ threshold {UNKNOWN_THRESHOLD})."
                )
                if not args.force:
                    ans = input("    Add as new wafer anyway? [y/N] ").strip().lower()
                    if ans != "y":
                        print("    Aborted.")
                        return
                else:
                    print("    (--force, skipping confirmation)")

    idx   = db.add_image(blob_id, args.image, rgb, gray, disk)
    blob  = db.get_blob(blob_id)
    n_tot = len(blob["images"])
    action = "Added another view to" if adding_to_existing else "Created new wafer"
    print(f"\n[+] {action} '{blob_id}'  [UUID: {blob['short_id']}]  —  "
          f"image #{idx}, {n_tot} view{'s' if n_tot != 1 else ''} total")


# ── query ─────────────────────────────────────────────────────────────────────

def cmd_query(args) -> None:
    if not os.path.isfile(args.image):
        sys.exit(f"[!] File not found: {args.image}")

    db = BlobDB(args.db)
    if not db.blob_ids():
        sys.exit("[!] Database is empty. Run 'add' first.")

    device = _device(getattr(args, "no_gpu", False))
    disk = _load_matcher(device)

    print(f"[*] Preprocessing {args.image} ...")
    proc = preprocess(args.image)
    gray = proc["gray"]
    print(f"    Cropped shape: {gray.shape[1]}×{gray.shape[0]} px")
    print(f"[*] Matching against {len(db.blob_ids())} blob(s) ...")

    verbose = getattr(args, "debug", False)
    fine_rotation = getattr(args, "fine_rotation", False)
    scores = run_matching(gray, db, disk, verbose=verbose,
                          fine_rotation=fine_rotation)
    _print_results(scores, top_k=args.top_k, threshold=args.threshold)


# ── list ──────────────────────────────────────────────────────────────────────

def cmd_list(args) -> None:
    db = BlobDB(args.db)
    blobs = db.blob_ids()
    if not blobs:
        print("[i] Database is empty.")
        return

    print(f"[i] {len(blobs)} wafer(s) in '{args.db}':\n")
    for blob_id in blobs:
        blob    = db.get_blob(blob_id)
        imgs    = blob["images"]
        short_id = blob.get("short_id", "????????")
        total_kp = sum(img.get("kp_count", 0) for img in imgs)
        print(f"  {blob_id:<22}  uuid:{short_id}  "
              f"{len(imgs)} view(s),  ~{total_kp} keypoints")
        for img in imgs:
            src = img.get("source_path", img.get("source", "?"))
            kp  = img.get("kp_count", "?")
            print(f"    [{img['img_idx']}]  {src}  (kp={kp})")
    print()


# ── compare ───────────────────────────────────────────────────────────────────

def cmd_compare(args) -> None:
    for p in (args.image0, args.image1):
        if not os.path.isfile(p):
            sys.exit(f"[!] File not found: {p}")

    device = _device(getattr(args, "no_gpu", False))
    disk = _load_matcher(device)

    print(f"[*] Preprocessing ...")
    proc0 = preprocess(args.image0)
    proc1 = preprocess(args.image1)
    print(f"    {os.path.basename(args.image0):30s}  {proc0['gray'].shape[1]}×{proc0['gray'].shape[0]}")
    print(f"    {os.path.basename(args.image1):30s}  {proc1['gray'].shape[1]}×{proc1['gray'].shape[0]}")

    # Rotation search: find best orientation of image1 relative to image0
    print(f"[*] Rotation search (DISK+LG) ...")
    feats0 = disk.extract(proc0["gray"])
    best_angle = 0.0
    best_disk  = 0
    for angle in ROTATION_ANGLES:
        gray1_rot  = _rotate_gray(proc1["gray"], angle)
        feats1_rot = disk.extract(gray1_rot)
        s = disk.match(feats0, feats1_rot)
        print(f"    @{angle:3d}°  DISK+LG={s}")
        if s > best_disk:
            best_disk  = s
            best_angle = float(angle)

    fine_rotation = getattr(args, "fine_rotation", False)
    if fine_rotation:
        print(f"[*] Fine rotation search around {best_angle:.0f}° ...")
        for offset in FINE_ROTATION_OFFSETS:
            fine_angle = best_angle + offset
            gray1_rot  = _rotate_gray(proc1["gray"], fine_angle)
            feats1_rot = disk.extract(gray1_rot)
            s = disk.match(feats0, feats1_rot)
            print(f"    @{fine_angle:+.1f}°  DISK+LG={s}")
            if s > best_disk:
                best_disk  = s
                best_angle = fine_angle

    angle_msg = f"{best_angle:+.0f}°" if best_angle != 0 else "0° (no rotation)"
    print(f"    Best: {angle_msg}  (DISK+LG={best_disk})")

    # Apply best rotation to image1, re-extract features
    rgb1_rot  = _rotate_rgb(proc1["rgb"], best_angle)
    gray1_rot = _rotate_gray(proc1["gray"], best_angle)
    feats1    = disk.extract(gray1_rot)
    print(f"    kp0={len(feats0['keypoints'])}  kp1={len(feats1['keypoints'])}")

    print(f"[*] Building comparison visualization ...")
    from visualizer import build_comparison, DISK_THRESHOLD
    canvas = build_comparison(
        proc0["rgb"], rgb1_rot,
        feats0, feats1,
        disk,
        path0=args.image0, path1=args.image1,
        query_rotation=best_angle,
    )

    out_path = args.output
    cv2.imwrite(out_path, canvas)
    print(f"[+] Saved → {out_path}  ({canvas.shape[1]}×{canvas.shape[0]} px)")

    n_disk     = disk.match(feats0, feats1)
    angle_note = f"  rotation: {best_angle:+.0f}°" if best_angle != 0 else ""

    print()
    print(f"  DISK+LG : {n_disk:5d} inliers  "
          f"({'SAME' if n_disk >= DISK_THRESHOLD else 'diff'}, thr={DISK_THRESHOLD}){angle_note}")
    print()
    if n_disk >= DISK_THRESHOLD:
        print(f"[✓] SAME BLOB")
    else:
        print(f"[✗] DIFFERENT")


# ── clear ─────────────────────────────────────────────────────────────────────

def cmd_clear(args) -> None:
    db = BlobDB(args.db)

    # Resolve wafer: accept --name or --wafer UUID
    blob_id = None
    if getattr(args, "wafer", None):
        blob_id = db.resolve_blob_id(args.wafer)
        if blob_id is None:
            sys.exit(f"[!] Wafer '{args.wafer}' not found. Run 'list' to see names and UUIDs.")
    elif getattr(args, "name", None):
        blob_id = args.name
        if db.get_blob(blob_id) is None:
            sys.exit(f"[!] Wafer '{blob_id}' not found in DB.")

    img_idx = getattr(args, "image", None)

    if blob_id and img_idx is not None:
        # Remove a single image from a wafer
        blob = db.get_blob(blob_id)
        valid = [img["img_idx"] for img in blob["images"]]
        if img_idx not in valid:
            sys.exit(f"[!] Image #{img_idx} not found in '{blob_id}'. Valid: {valid}")
        if not args.yes:
            src = next(i["source_path"] for i in blob["images"] if i["img_idx"] == img_idx)
            ans = input(f"Remove image #{img_idx} ({os.path.basename(src)}) "
                        f"from '{blob_id}'? [y/N] ").strip().lower()
            if ans != "y":
                print("Aborted.")
                return
        db.remove_image(blob_id, img_idx)
        remaining = db.get_blob(blob_id)
        if remaining:
            print(f"[+] Removed image #{img_idx} from '{blob_id}'  "
                  f"({len(remaining['images'])} view(s) remaining).")
        else:
            print(f"[+] Removed image #{img_idx} — '{blob_id}' has no views left and was deleted.")

    elif blob_id:
        # Remove entire wafer
        blob = db.get_blob(blob_id)
        if not args.yes:
            ans = input(f"Remove wafer '{blob_id}' [{blob['short_id']}] "
                        f"and all {len(blob['images'])} view(s)? [y/N] ").strip().lower()
            if ans != "y":
                print("Aborted.")
                return
        db.remove_blob(blob_id)
        print(f"[+] Removed wafer '{blob_id}'.")

    elif args.all:
        if not args.yes:
            ans = input(f"Clear ENTIRE database '{args.db}'? [y/N] ").strip().lower()
            if ans != "y":
                print("Aborted.")
                return
        db.clear()
        print(f"[+] Database '{args.db}' cleared.")

    else:
        sys.exit("[!] Specify --name NAME, --wafer UUID, or --all.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wafer blob fingerprint identification  (LoFTR + DISK/LightGlue)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db", default=DEFAULT_DB,
                        help=f"Database directory  (default: {DEFAULT_DB})")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU inference (ignore CUDA even if available)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-image match scores")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- add ---
    p_add = sub.add_parser("add", help="Add a reference image to the DB")
    p_add.add_argument("image",
                       help="Path to image (TIF, JPEG, PNG, BMP)")
    add_id = p_add.add_mutually_exclusive_group(required=True)
    add_id.add_argument("--new",
                        help="Create a new wafer with this identifier, e.g. wafer1")
    add_id.add_argument("--wafer",
                        help="Wafer name or 4-char UUID of existing wafer to add a new view to")
    p_add.add_argument("--force", action="store_true",
                       help="Skip duplicate-detection warning")

    # --- query ---
    p_query = sub.add_parser("query", help="Identify an unknown image")
    p_query.add_argument("image",
                         help="Path to query image (TIF, JPEG, PNG, BMP)")
    p_query.add_argument("--top-k", type=int, default=5,
                         help="Number of ranked results to show  (default: 5)")
    p_query.add_argument("--threshold", type=int, default=UNKNOWN_THRESHOLD,
                         help=f"Min inliers for a positive match  "
                              f"(default: {UNKNOWN_THRESHOLD})")
    p_query.add_argument("--debug", action="store_true",
                         help="Print per-image match scores")
    p_query.add_argument("--fine-rotation", action="store_true",
                         help="Search ±5/10/15° around the best 90°-step rotation")

    # --- compare ---
    p_cmp = sub.add_parser("compare",
                           help="Visually compare two images: are they the same blob?")
    p_cmp.add_argument("image0", help="First image (TIF, JPEG, PNG, BMP)")
    p_cmp.add_argument("image1", help="Second image (TIF, JPEG, PNG, BMP)")
    p_cmp.add_argument("--output", "-o", default="compare_result.png",
                       help="Output PNG path  (default: compare_result.png)")
    p_cmp.add_argument("--fine-rotation", action="store_true",
                       help="Search ±5/10/15° around the best 90°-step rotation")

    # --- list ---
    sub.add_parser("list", help="List all blobs in the DB")

    # --- clear ---
    p_clear = sub.add_parser("clear", help="Remove blob(s) from the DB")
    p_clear.add_argument("--name",
                         help="Blob identifier to remove entirely")
    p_clear.add_argument("--wafer",
                         help="Wafer name or 4-char UUID to remove (or to remove a single image from)")
    p_clear.add_argument("--image", type=int, metavar="N",
                         help="Image index to remove (use with --wafer; see 'list' for indices)")
    p_clear.add_argument("--all", action="store_true",
                         help="Wipe the entire database")
    p_clear.add_argument("--yes", "-y", action="store_true",
                         help="Skip confirmation prompt")

    args = parser.parse_args()
    {"add": cmd_add, "query": cmd_query, "compare": cmd_compare,
     "list": cmd_list, "clear": cmd_clear}[args.cmd](args)


if __name__ == "__main__":
    main()
