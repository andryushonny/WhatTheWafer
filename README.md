# WhatTheWafer

<table border="0"><tr>
<td><img src="icon.png" width="160" alt="WhatTheWafer logo"/></td>
<td align="center"><strong>SILICON WAFER IDENTIFICATION BY UNIQUE BLOB FINGERPRINT</strong></td>
</tr></table>

Each silicon wafer carries a unique feature in the form of a marker blob (which helps ensure proper orientation) in a corner, with a distinctive pattern of interference fringes. The system memorizes this "digital fingerprint" and records it in a database. Subsequently, from a single image, it determines which wafer it is and what data is stored about it in the database.

```
whattw query wafer_photo.jpg

  Rank  Blob ID                   Inliers    Rotation
  ────────────────────────────────────────────────────
  1     sample_42                    4764          0°  ◀
  2     sample_07                      18        +90°
  3     sample_15                       9          0°

[✓] MATCH → 'sample_42'  (4764 matches)
```

---

## How it works

The system does not retrain when new wafers are added. Instead it uses two pretrained computer vision models:

- **LoFTR** — finds a dense grid of matching points between two images
- **DISK + LightGlue** — detects keypoints and matches them individually

When a wafer is added (`add`), the models extract a "digital fingerprint" of the image once and store it in the database. On a query (`query`), the fingerprint of the new image is compared against the database — the candidate with the most geometrically consistent matches wins.

The system automatically checks all 90° rotations — if the wafer was photographed at a different orientation than when it was added, that is not a problem.

---

## Installation

**Requirements:** Python 3.10+, pip

```bash
cd WhatTheWafer

# Install dependencies and register the whattw command
pip install -e .
```

An NVIDIA GPU will be used automatically if available. The system also runs on CPU, but significantly slower.

> Use [QUICKSTART.md](QUICKSTART.md) to get up to speed quickly.

---

## Quick start

### Step 1 — Add reference images

Photograph each wafer 2–4 times (different angles, different lighting) and add them to the database:

```bash
whattw add photo1.tif --new sample_42
whattw add photo2.tif --wafer sample_42   # second view of the same wafer
whattw add other.tif  --new sample_07
```

Accepted formats: **TIF, JPEG, PNG, BMP**.

### Step 2 — Identify an unknown image

```bash
whattw query unknown.jpg
```

### Step 3 — Visually verify the match

```bash
whattw compare reference.tif unknown.jpg --output result.png
```

This saves a three-panel image: DISK+LightGlue matches, LoFTR matches, and the verdict.

![Comparison example](compare_same.png)

---

## All commands

### `add` — add a wafer to the database

```bash
whattw add <image> --new <name>       # create a new wafer
whattw add <image> --wafer <name|uuid>  # add a view to an existing wafer
```

| Flag | Description |
|---|---|
| `--new NAME` | Create a new wafer with this identifier, e.g. `sample_42` |
| `--wafer NAME\|UUID` | Wafer name or 4-char UUID to add a new view to |
| `--force` | Skip duplicate-detection warning |
| `--matcher disk\|loftr\|both` | Matcher for duplicate check (default: `disk`) |

### `query` — identify an image

```bash
whattw query <image>
```

| Flag | Description |
|---|---|
| `--top-k N` | Show N best matches (default: 5) |
| `--threshold N` | Minimum inlier count for a positive match (default: 15) |
| `--matcher disk\|loftr\|both` | Matcher to use (default: `both`) |
| `--fine-rotation` | Extra search at ±5/10/15° around the best 90°-step rotation |
| `--debug` | Print per-image match scores |

### `compare` — visually compare two images

```bash
whattw compare <image1> <image2> --output result.png
```

| Flag | Description |
|---|---|
| `--output`, `-o` | Output PNG path (default: `compare_result.png`) |
| `--fine-rotation` | Fine rotation search |

### `list` — list all wafers in the database

```bash
whattw list
```

### `clear` — remove from the database

```bash
whattw clear --wafer sample_42          # remove one wafer (by name or UUID)
whattw clear --wafer a3f1 --image 1     # remove a single view (see index in list)
whattw clear --all                      # wipe the entire database
```

### Global flag `--no-gpu`

```bash
whattw --no-gpu query unknown.jpg
```

Forces CPU inference even if a GPU is available.

---

## Tips

| Situation | Recommendation |
|---|---|
| Dark-field imaging | Add at least one dark-field image of the same wafer |
| Very close-up frame | Add a close-up shot as an additional reference |
| Image slightly tilted (< 15°) | Use `--fine-rotation` |
| Wafer not found | Run with `--debug` to see per-reference match scores |
| False duplicate warning on `add` | Use `--force` to add anyway |

**How many reference images are needed?**  
Minimum 1, optimal 2–4. Add images under varied conditions — lighting, scale, orientation. The more diverse the references, the more robust the identification.

---

## Project structure

```
WhatTheWafer/
├── wafer_id.py          — entry point, CLI
├── preprocessing.py     — image loading and blob segmentation
├── database.py          — storage (SQLite + HDF5 + FAISS)
├── visualizer.py        — match visualization
├── matchers/
│   ├── loftr_matcher.py          — dense LoFTR matcher
│   └── disk_lightglue_matcher.py — keypoint DISK+LightGlue matcher
├── models/
│   └── checkpoints/     — model weights (~95 MB, bundled with the project)
├── database/            — wafer database (created automatically on first add)
├── Dockerfile           — for building a portable image
└── DEPLOY.md            — deployment guide for an offline PC
```

---

## Technical details

| | |
|---|---|
| Models | LoFTR (indoor), DISK (depth), LightGlue |
| Storage | SQLite (metadata) + HDF5 (features) + FAISS FlatL2 (fast retrieval) |
| Input formats | TIF (16-bit, OME), JPEG, PNG, BMP |
| Query speed | ~0.5 s on GPU regardless of database size (FAISS pre-filters candidates) |
| Retraining on `add` | None — models are frozen; `add` only runs forward inference |
