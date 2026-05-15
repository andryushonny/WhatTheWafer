"""
wafer_id.py — Wafer blob fingerprint identification (DISK + LightGlue)

Commands:
  add    Add a reference image to the database
  query  Identify an unknown image
  list   List all blobs in the database
  clear  Remove a blob (or the whole DB)
  compare Visually compare two images

Examples:
  python wafer_id.py add wafers/wafer1_1.tif --new wafer1
  python wafer_id.py add wafers/wafer1_2.tif --wafer wafer1
  python wafer_id.py add wafers/wafer2_1.tif --new wafer2
  python wafer_id.py query wafers/wafer1_3.tif
  python wafer_id.py query wafers/wafer1_3.tif --debug
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


# ── rotation + batch-extraction helpers ──────────────────────────────────────

def _extract_cardinal_rotations(
    disk: DISKLightGlueMatcher, gray: np.ndarray
) -> dict[float, dict]:
    """Extract DISK features for all 4 cardinal rotations using 2 batch forward passes.

    0°+180° share spatial dimensions (H, W); 90°+270° share (W, H).
    Two batch calls instead of four sequential calls → ~2x faster on GPU,
    ~2x faster on CPU for the extraction stage.
    """
    gray_0   = gray                           # 0°  : (H, W)
    gray_90  = _rotate_gray(gray, 90)         # 90° : (W, H)
    gray_180 = _rotate_gray(gray, 180)        # 180°: (H, W)  same as 0°
    gray_270 = _rotate_gray(gray, 270)        # 270°: (W, H)  same as 90°
    f_0_180  = disk.extract_batch([gray_0, gray_180])
    f_90_270 = disk.extract_batch([gray_90, gray_270])
    return {
        0.0:   f_0_180[0],
        90.0:  f_90_270[0],
        180.0: f_0_180[1],
        270.0: f_90_270[1],
    }


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

    All rotation features are extracted ONCE before the candidate loop so that
    disk.extract() is never called inside the per-blob matching logic.

    Returns list of (blob_id, score, best_angle_degrees) sorted descending.
    """
    # Pre-extract features for all needed angles before the candidate loop.
    # Cardinal rotations: 2 batch forward passes (0+180, 90+270) instead of 4 sequential.
    # Fine rotations: individual calls, but only executed once here (not per-blob).
    q_disk_by_angle: dict[float, dict] = _extract_cardinal_rotations(disk, gray_query)
    if fine_rotation:
        for base in ROTATION_ANGLES:
            for offset in FINE_ROTATION_OFFSETS:
                a = float(base + offset)
                if a not in q_disk_by_angle:
                    q_disk_by_angle[a] = disk.extract(_rotate_gray(gray_query, a))

    # FAISS first stage: narrow to a short-list of candidates (cardinal angles only)
    q_descs_cardinal = {
        a: q["descriptors"] for a, q in q_disk_by_angle.items()
        if a in {0.0, 90.0, 180.0, 270.0}
    }
    candidate_ids = db.fast_candidates(q_descs_cardinal, top_k=FAST_TOP_K)
    if verbose:
        print(f"  [FAISS] candidates: {candidate_ids}")

    all_feats = db.load_blob_features(candidate_ids)
    per_blob: dict[str, tuple[int, float]] = {}

    for blob_id, ref_list in all_feats.items():
        best_score = 0
        best_angle = 0.0
        for i, ref in enumerate(ref_list):
            img_best_score = 0
            img_best_angle = 0.0

            # Cardinal search
            for angle in ROTATION_ANGLES:
                s = disk.match(ref["disk"], q_disk_by_angle[float(angle)])
                if verbose:
                    print(f"    {blob_id}[{i}]  DISK+LG@{angle:3d}°={s}")
                if s > img_best_score:
                    img_best_score = s
                    img_best_angle = float(angle)

            # Fine-rotation search around the best cardinal (features already extracted)
            if fine_rotation:
                for offset in FINE_ROTATION_OFFSETS:
                    fine_angle = img_best_angle + offset
                    s = disk.match(ref["disk"], q_disk_by_angle[fine_angle])
                    if verbose:
                        print(f"    {blob_id}[{i}]  DISK+LG@{fine_angle:+.0f}°={s}")
                    if s > img_best_score:
                        img_best_score = s
                        img_best_angle = fine_angle

            if img_best_score > best_score:
                best_score = img_best_score
                best_angle = img_best_angle

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
    images: list[str] = args.image   # always a list (nargs='+')
    for p in images:
        if not os.path.isfile(p):
            sys.exit(f"[!] File not found: {p}")

    db = BlobDB(args.db)

    # Resolve target blob_id
    if args.wafer:
        blob_id = db.resolve_blob_id(args.wafer)
        if blob_id is None:
            sys.exit(f"[!] Wafer '{args.wafer}' not found. Run 'list' to see names and UUIDs.")
        adding_to_existing = True
    else:
        blob_id = args.new
        if db.get_blob(blob_id) is not None:
            existing = db.get_blob(blob_id)
            sys.exit(
                f"[!] Wafer '{blob_id}' already exists "
                f"(uuid:{existing['short_id']}).\n"
                f"    To add another view use:  whattw add {images[0]} --wafer {blob_id}"
            )
        adding_to_existing = False

    device = _device(getattr(args, "no_gpu", False))
    disk = _load_matcher(device)

    # Preprocess and optionally duplicate-check the first image before the batch
    cached_procs: dict[str, dict] = {}
    if not adding_to_existing and db.blob_ids():
        print(f"[*] Checking against {len(db.blob_ids())} existing blob(s) ...")
        first_proc = preprocess(images[0])
        cached_procs[images[0]] = first_proc
        scores = run_matching(first_proc["gray"], db, disk, verbose=args.debug)
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

    # Add all images; FAISS is rebuilt once after the batch, not after each image
    n_added = 0
    last_idx = 0
    with db.batch_add():
        for img_path in images:
            print(f"[*] Preprocessing {img_path} ...")
            proc = cached_procs.get(img_path) or preprocess(img_path)
            rgb, gray = proc["rgb"], proc["gray"]
            print(f"    Cropped shape: {gray.shape[1]}×{gray.shape[0]} px")
            last_idx = db.add_image(blob_id, img_path, rgb, gray, disk)
            n_added += 1

    blob  = db.get_blob(blob_id)
    n_tot = len(blob["images"])
    action = "Added view(s) to" if adding_to_existing else "Created new wafer"
    views_word = "view" if n_added == 1 else "views"
    print(f"\n[+] {action} '{blob_id}'  [UUID: {blob['short_id']}]  —  "
          f"{n_added} {views_word} added, {n_tot} total")


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

    # Rotation search: extract all 4 cardinal orientations of image1 in 2 batch
    # forward passes, then evaluate each against image0.
    print(f"[*] Rotation search (DISK+LG) ...")
    feats0         = disk.extract(proc0["gray"])
    feats1_by_angle = _extract_cardinal_rotations(disk, proc1["gray"])

    best_angle = 0.0
    best_disk  = 0
    best_feats1: dict       = feats1_by_angle[0.0]
    best_rgb1:  np.ndarray  = proc1["rgb"]

    for angle in ROTATION_ANGLES:
        s = disk.match(feats0, feats1_by_angle[float(angle)])
        print(f"    @{angle:3d}°  DISK+LG={s}")
        if s > best_disk:
            best_disk   = s
            best_angle  = float(angle)
            best_feats1 = feats1_by_angle[float(angle)]
            best_rgb1   = _rotate_rgb(proc1["rgb"], angle)

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
                best_disk   = s
                best_angle  = fine_angle
                best_feats1 = feats1_rot
                best_rgb1   = _rotate_rgb(proc1["rgb"], fine_angle)

    angle_msg = f"{best_angle:+.0f}°" if best_angle != 0 else "0° (no rotation)"
    print(f"    Best: {angle_msg}  (DISK+LG={best_disk})")

    feats1   = best_feats1
    rgb1_rot = best_rgb1
    print(f"    kp0={len(feats0['keypoints'])}  kp1={len(feats1['keypoints'])}")

    print(f"[*] Building comparison visualization ...")
    from visualizer import build_comparison, DISK_THRESHOLD
    canvas, n_disk = build_comparison(
        proc0["rgb"], rgb1_rot,
        feats0, feats1,
        disk,
        path0=args.image0, path1=args.image1,
        query_rotation=best_angle,
    )

    out_path = args.output
    cv2.imwrite(out_path, canvas)
    print(f"[+] Saved → {out_path}  ({canvas.shape[1]}×{canvas.shape[0]} px)")

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
        description="Wafer blob fingerprint identification  (DISK + LightGlue)",
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
    p_add.add_argument("image", nargs='+',
                       help="Path(s) to image (TIF, JPEG, PNG, BMP); "
                            "multiple paths add several views in one FAISS rebuild")
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
