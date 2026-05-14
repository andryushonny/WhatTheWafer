# WhatTheWafer — Quick Start

## Installation

```bash
cd WhatTheWafer
pip install -e .
```

GPU (CUDA) is used automatically when available; CPU works but is ~5× slower.

---

## Workflow

### 1. Add reference images

```bash
whattw add wafers/wafer1_1.tif --new wafer1
whattw add wafers/wafer1_2.tif --wafer wafer1   # second view of the same wafer
whattw add wafers/wafer2_1.tif --new wafer2
```

Add **2–4 images per wafer** for best robustness — different lighting, zoom, or orientation.  
Accepts TIF, JPEG, PNG, BMP.

### 2. Identify an unknown image

```bash
whattw query wafers/unknown.tif
```

All four 90° rotations are tried automatically.

```
  Rank  Blob ID                   Inliers    Rotation
  ------------------------------------------------------
  1     wafer1                       5499          0°  ◀
  2     wafer3                         17          0°
  3     wafer2                         10        +90°

[✓] MATCH → 'wafer1'  (5499 inliers)
```

### 3. Visual comparison

```bash
whattw compare wafers/wafer1_1.tif wafers/unknown.tif -o result.png
```

Produces a 3-panel PNG: DISK+LightGlue matches, LoFTR matches, verdict bar.  
The second image is auto-rotated to align with the first.

### 4. Manage the database

```bash
whattw list                            # show all wafers with UUIDs and view counts
whattw clear --wafer wafer1            # remove one wafer (by name or 4-char UUID)
whattw clear --wafer wafer1 --image 1  # remove a single view (index from list)
whattw clear --all                     # wipe everything
```

---

## Tips

| Situation | What to do |
|---|---|
| Dark-field images | Add at least one dark-field reference per wafer |
| Very zoomed-in shot | Add a close-up reference image |
| Still not matching | Run with `--debug` to see per-image scores |
| Tilted shot (< 15°) | Add `--fine-rotation` to the query |
| False positive warning on `add` | Use `--force` to skip the confirmation |

---

## Key options

```
add     --new NAME                           (create new wafer)
        --wafer NAME|UUID                    (add view to existing wafer)
        --matcher      disk | loftr | both   (duplicate check, default: disk)
        --force                              (skip duplicate warning)

query   --matcher      loftr | disk | both   (default: both)
        --top-k        N                     (default: 5)
        --threshold    N                     (min inliers, default: 15)
        --fine-rotation                      (also search ±5/10/15°)
        --debug                              (verbose per-image scores)

compare --output PATH                        (default: compare_result.png)
        --fine-rotation

Global  --no-gpu                             (force CPU inference)
```