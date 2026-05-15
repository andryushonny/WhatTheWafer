"""
crossval.py — Random cross-validation / sanity check for WhatTheWafer.

For images already in the database: leave-one-out (LOO) — temporarily removes
the image, queries, then restores it. This ensures no self-match inflates scores.
For images NOT in the database: queried normally.
Images that are the only view in their blob are skipped (nothing left to match).

Ground truth is extracted from filenames that follow the pattern
<letters><digits>_<anything>  (e.g. wafer1_3.tif → group "wafer1").

Usage:
  whattw xval
  whattw xval --wafers-dir wafers/ --n 5 --seed 123
  python crossval.py --wafers-dir wafers/ --n 10 --seed 42
"""

import argparse
import hashlib
import os
import random
import re
import sys

from preprocessing import preprocess, extract_metadata
from database import BlobDB

IMAGE_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}
_GT_RE = re.compile(r"^([a-zA-Z]+\d+)_")


def _extract_gt(filename: str) -> str | None:
    m = _GT_RE.match(os.path.basename(filename))
    return m.group(1) if m else None


def _path_index(db: BlobDB) -> dict[str, tuple[str, int]]:
    """Return {normalized_abs_path: (blob_id, img_idx)} for images whose source file still exists."""
    index: dict[str, tuple[str, int]] = {}
    for blob_id in db.blob_ids():
        blob = db.get_blob(blob_id)
        for img in blob["images"]:
            src = img.get("source_path", "")
            if src:
                norm = os.path.normpath(os.path.abspath(src))
                if os.path.isfile(norm):
                    index[norm] = (blob_id, img["img_idx"])
    return index


def _image_hash(rgb) -> str:
    return hashlib.sha256(rgb.tobytes()).hexdigest()[:16]


def cmd_xval(args) -> None:
    from wafer_id import UNKNOWN_THRESHOLD, _device, _load_matcher, run_matching

    wafers_dir = args.wafers_dir
    if not os.path.isdir(wafers_dir):
        sys.exit(f"[!] Directory not found: {wafers_dir}")

    all_images = sorted(
        os.path.join(wafers_dir, fn)
        for fn in os.listdir(wafers_dir)
        if os.path.splitext(fn)[1].lower() in IMAGE_EXTS
    )
    if not all_images:
        sys.exit(f"[!] No images found in '{wafers_dir}'.")

    db = BlobDB(args.db)
    if not db.blob_ids():
        sys.exit("[!] Database is empty. Run 'whattw add' first.")

    threshold = getattr(args, "threshold", UNKNOWN_THRESHOLD)
    disk = _load_matcher(_device(getattr(args, "no_gpu", False)))
    verbose = getattr(args, "debug", False)

    hash_idx = db.hash_index()          # {image_hash: (blob_id, img_idx)}  — primary
    path_idx = _path_index(db)          # {norm_abs_path: (blob_id, img_idx)} — fallback

    n = min(args.n, len(all_images))
    if args.n > len(all_images):
        print(f"[!] --n={args.n} exceeds available images ({len(all_images)}); sampling all.")

    sampled = random.Random(args.seed).sample(all_images, n)

    CF, CE, CP = 22, 9, 20
    header = (f" {'#':>2} │ {'File':<{CF}} │ {'Expected':<{CE}} │"
              f" {'Predicted':<{CP}} │ {'Score':>5} │ {'OK?':^5} │ Mode")
    div = "─" * len(header)

    print(f"\n=== WhatTheWafer xval ===")
    print(f"Wafers dir : {wafers_dir}    DB : {args.db}")
    print(f"Sampled    : {n} / {len(all_images)}    Seed: {args.seed}    Threshold: {threshold}")
    print()
    print(header)
    print(div)

    results: list[dict] = []
    for idx, img_path in enumerate(sampled, 1):
        filename = os.path.basename(img_path)
        expected = _extract_gt(filename)
        norm_path = os.path.normpath(os.path.abspath(img_path))

        try:
            proc = preprocess(img_path)
        except Exception as exc:
            print(f" {idx:>2} │ {filename[:CF]:<{CF}} │ {(expected or '—'):<{CE}} │"
                  f" {str(exc)[:CP]:<{CP}} │ {'—':>5} │ {'—':^5} │ ERROR")
            results.append({"error": True, "skipped": False})
            continue

        # Detect if this image is already in the DB:
        # prefer hash lookup (content-based, path-independent),
        # fall back to path lookup for older entries without a stored hash.
        h = _image_hash(proc["rgb"])
        if h in hash_idx:
            blob_id, img_idx = hash_idx[h]
            in_db = True
        elif norm_path in path_idx:
            blob_id, img_idx = path_idx[norm_path]
            in_db = True
        else:
            in_db = False

        if in_db:
            blob = db.get_blob(blob_id)
            if len(blob["images"]) <= 1:
                print(f" {idx:>2} │ {filename[:CF]:<{CF}} │ {(expected or '—'):<{CE}} │"
                      f" {'—':<{CP}} │ {'—':>5} │ {'—':^5} │ SKIP (only view)")
                results.append({"error": False, "skipped": True, "expected": expected})
                continue
            mode = "LOO"
            db.remove_image(blob_id, img_idx)
        else:
            mode = "new"

        scores = run_matching(proc["gray"], db, disk, verbose=verbose)

        if in_db:
            db.add_image(blob_id, img_path, proc["rgb"], proc["gray"], disk,
                         metadata=extract_metadata(img_path))
            # Refresh indices after LOO restore (img_idx may have changed)
            hash_idx = db.hash_index()
            path_idx = _path_index(db)

        if not scores:
            score, above, predicted_id, predicted_display = 0, False, None, "—"
        else:
            predicted_id, score, _ = scores[0]
            above = score >= threshold
            blob_info = db.get_blob(predicted_id)
            short = blob_info["short_id"] if blob_info else "????"
            tag = f"({short})"
            name = predicted_id[:CP - len(tag)] if len(predicted_id) + len(tag) > CP else predicted_id
            predicted_display = f"{name}{tag}" if above else f"({predicted_id[:CP - 2]})"

        if expected is not None and predicted_id is not None:
            correct: bool | None = above and (predicted_id == expected)
        else:
            correct = None

        ok_str = ("✓" if correct else "✗") if correct is not None else ("△" if above else "—")

        print(f" {idx:>2} │ {filename[:CF]:<{CF}} │ {(expected or '—'):<{CE}} │"
              f" {predicted_display[:CP]:<{CP}} │ {score:>5} │ {ok_str:^5} │ {mode}")

        results.append({
            "error": False, "skipped": False,
            "expected": expected, "predicted_id": predicted_id,
            "score": score, "above": above, "correct": correct,
        })

    # ── summary ───────────────────────────────────────────────────────────────
    tested = [r for r in results if not r["error"] and not r.get("skipped")]
    skipped_n = sum(1 for r in results if r.get("skipped"))
    scores_list = [r["score"] for r in tested]
    above_n = sum(1 for r in tested if r["above"])
    gt_results = [r for r in tested if r["expected"] is not None]
    correct_n = sum(1 for r in gt_results if r["correct"])

    print(f"\n=== Summary ===")
    if skipped_n:
        print(f"Skipped (only view)    : {skipped_n}  — add more photos of those wafers")
    if gt_results:
        pct = 100 * correct_n / len(gt_results)
        print(f"Ground truth available : {len(gt_results)} / {len(tested)}")
        print(f"Accuracy (GT images)   : {correct_n} / {len(gt_results)} = {pct:.1f}%")
    else:
        print(f"Ground truth available : 0 / {len(tested)}  (no parseable filenames — accuracy N/A)")

    if scores_list:
        pct_above = 100 * above_n / len(tested)
        mean = sum(scores_list) / len(scores_list)
        print(f"Above threshold        : {above_n} / {len(tested)} = {pct_above:.1f}%")
        print(f"Mean score             : {mean:.1f}   Min: {min(scores_list)}   Max: {max(scores_list)}")
    print()


def main() -> None:
    from wafer_id import UNKNOWN_THRESHOLD, DEFAULT_DB

    parser = argparse.ArgumentParser(
        description="Random cross-validation sanity check for WhatTheWafer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--wafers-dir", default="wafers", dest="wafers_dir",
                        help="Directory with wafer images to sample from  (default: wafers)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of images to sample  (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility  (default: 42)")
    parser.add_argument("--threshold", type=int, default=UNKNOWN_THRESHOLD,
                        help=f"Min inliers for a positive match  (default: {UNKNOWN_THRESHOLD})")
    parser.add_argument("--db", default=DEFAULT_DB,
                        help=f"Database directory  (default: {DEFAULT_DB})")
    parser.add_argument("--no-gpu", action="store_true", dest="no_gpu",
                        help="Force CPU inference")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-image match scores")
    cmd_xval(parser.parse_args())


if __name__ == "__main__":
    main()
