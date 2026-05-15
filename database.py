"""
database.py — Blob fingerprint database.

Storage layout (all in db_dir):
  index.sqlite   — metadata: blobs + images tables
  features.h5    — all feature arrays, keyed by blobs/<blob_id>/<NNN>/
  faiss.index    — FAISS ScalarQuantizer(QT_8bit) index over L2-normalised 128-d DISK descriptors
  faiss_map.npy  — int32 array: faiss_vector_id → index into faiss_blob_ids
  faiss_ids.json — ordered list of blob_ids corresponding to faiss_map values
  <blob_id>/thumb.jpg — representative thumbnail
"""

import json
import sqlite3
import shutil
import uuid
from contextlib import contextmanager
import cv2
import numpy as np
import h5py
import faiss  # type: ignore[import-untyped]
from datetime import datetime, timezone
from pathlib import Path

THUMB_SIZE       = 256
FAISS_CANDIDATES = 5   # blobs to short-list in the fast first stage


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_short_id() -> str:
    return uuid.uuid4().hex[:4]


class BlobDB:
    def __init__(self, db_dir: str = "database"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self._sqlite_path = self.db_dir / "index.sqlite"
        self._h5_path     = self.db_dir / "features.h5"
        self._faiss_path  = self.db_dir / "faiss.index"
        self._fmap_path   = self.db_dir / "faiss_map.npy"
        self._fids_path   = self.db_dir / "faiss_ids.json"

        self._conn = self._init_sqlite()
        self._faiss_index, self._faiss_map, self._faiss_blob_ids = self._load_faiss()
        self._batch_mode = False   # deferred FAISS rebuild flag for batch_add()

        # One-shot migration from the old folder-based format
        legacy = self.db_dir / "index.json"
        if legacy.exists():
            self._migrate_legacy(legacy)

        # One-shot HDF5 migrations (order matters: strip LoFTR first, then convert dtype)
        self._migrate_strip_loftr()
        self._migrate_to_float16()

        # Upgrade FAISS index type if the on-disk index is the old IndexFlatL2
        if isinstance(self._faiss_index, faiss.IndexFlat):
            print("[*] Upgrading FAISS: FlatL2 → ScalarQuantizer(QT_8bit) ...")
            self._rebuild_faiss()

    # ── SQLite ────────────────────────────────────────────────────────────────

    def _init_sqlite(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._sqlite_path))
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS blobs (
                blob_id  TEXT PRIMARY KEY,
                name     TEXT NOT NULL,
                added_at TEXT NOT NULL,
                short_id TEXT UNIQUE
            );
            CREATE TABLE IF NOT EXISTS images (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                blob_id     TEXT    NOT NULL,
                img_idx     INTEGER NOT NULL,
                source_path TEXT    NOT NULL,
                kp_count    INTEGER DEFAULT 0,
                thumb_path  TEXT,
                FOREIGN KEY (blob_id) REFERENCES blobs(blob_id)
            );
        """)
        conn.commit()

        # Add short_id column to existing DBs that predate this feature
        try:
            conn.execute("ALTER TABLE blobs ADD COLUMN short_id TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

        # Assign short_ids to any blobs that don't have one yet
        nulls = conn.execute(
            "SELECT blob_id FROM blobs WHERE short_id IS NULL"
        ).fetchall()
        for row in nulls:
            sid = self._unique_short_id(conn)
            conn.execute(
                "UPDATE blobs SET short_id=? WHERE blob_id=?", (sid, row["blob_id"])
            )
        if nulls:
            conn.commit()

        return conn

    @staticmethod
    def _unique_short_id(conn: sqlite3.Connection) -> str:
        """Generate a short_id that doesn't already exist in the DB."""
        while True:
            sid = _generate_short_id()
            if not conn.execute(
                "SELECT 1 FROM blobs WHERE short_id=?", (sid,)
            ).fetchone():
                return sid

    # ── FAISS ─────────────────────────────────────────────────────────────────

    def _load_faiss(self):
        if self._faiss_path.exists():
            index = faiss.read_index(str(self._faiss_path))
            fmap  = np.load(str(self._fmap_path)) if self._fmap_path.exists() \
                    else np.array([], dtype=np.int32)
            fids  = json.loads(self._fids_path.read_text()) \
                    if self._fids_path.exists() else []
        else:
            index = faiss.IndexScalarQuantizer(
                128, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2
            )
            fmap  = np.array([], dtype=np.int32)
            fids  = []
        return index, fmap, fids

    def _save_faiss(self) -> None:
        faiss.write_index(self._faiss_index, str(self._faiss_path))
        np.save(str(self._fmap_path), self._faiss_map)
        self._fids_path.write_text(json.dumps(self._faiss_blob_ids))

    @contextmanager
    def batch_add(self):
        """Defer FAISS rebuild until all add_image() calls complete.

        Use when adding multiple images to avoid rebuilding the index on every
        add — the index is rebuilt exactly once when the context exits.

        Example::

            with db.batch_add():
                for path in image_paths:
                    db.add_image(blob_id, path, rgb, gray, disk)
        """
        self._batch_mode = True
        try:
            yield self
        finally:
            self._batch_mode = False
            self._rebuild_faiss()

    def _rebuild_faiss(self) -> None:
        """Rebuild FAISS ScalarQuantizer index from scratch."""
        self._faiss_map      = np.array([], dtype=np.int32)
        self._faiss_blob_ids = []
        sq = faiss.IndexScalarQuantizer(
            128, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2
        )

        if not self._h5_path.exists():
            self._faiss_index = sq
            self._save_faiss()
            return

        all_descs: list[np.ndarray] = []
        blob_map:  list[int]        = []

        with h5py.File(str(self._h5_path), "r") as f:
            for blob_id in f.get("blobs", {}).keys():
                blob_idx = len(self._faiss_blob_ids)
                self._faiss_blob_ids.append(blob_id)
                for img_key in f["blobs"][blob_id].keys():
                    raw = f["blobs"][blob_id][img_key]["disk_desc"][:].astype(np.float32)
                    norms = np.linalg.norm(raw, axis=1, keepdims=True)
                    d = raw / (norms + 1e-6)
                    all_descs.append(d)
                    blob_map.extend([blob_idx] * len(d))

        if not all_descs:
            self._faiss_index = sq
            self._save_faiss()
            return

        all_vecs = np.vstack(all_descs)
        self._faiss_map = np.array(blob_map, dtype=np.int32)
        sq.train(all_vecs)
        sq.add(all_vecs)
        self._faiss_index = sq
        self._save_faiss()

    # ── HDF5 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _h5_key(blob_id: str, img_idx: int) -> str:
        return f"blobs/{blob_id}/{img_idx:03d}"

    def _h5_save(self, blob_id: str, img_idx: int, disk_feats: dict) -> None:
        with h5py.File(str(self._h5_path), "a") as f:
            g = f.require_group(self._h5_key(blob_id, img_idx))
            g.create_dataset("disk_kp",     data=disk_feats["keypoints"])
            g.create_dataset("disk_desc",   data=disk_feats["descriptors"].astype(np.float16))
            g.create_dataset("disk_scores", data=disk_feats["scores"])
            g.attrs["hw"] = disk_feats["hw"]

    def _h5_load(self, blob_id: str, img_idx: int) -> dict:
        key = self._h5_key(blob_id, img_idx)
        with h5py.File(str(self._h5_path), "r") as f:
            if key not in f:
                raise KeyError(f"Features not found in HDF5: {key}")
            g = f[key]
            return {
                "disk": {
                    "keypoints":   g["disk_kp"][:],
                    "descriptors": g["disk_desc"][:].astype(np.float32),
                    "scores":      g["disk_scores"][:],
                    "hw":          tuple(int(x) for x in g.attrs["hw"]),
                },
            }

    # ── public read API ───────────────────────────────────────────────────────

    def blob_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT blob_id FROM blobs ORDER BY added_at"
        ).fetchall()
        return [r["blob_id"] for r in rows]

    def get_blob(self, blob_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM blobs WHERE blob_id=?", (blob_id,)
        ).fetchone()
        if row is None:
            return None
        imgs = self._conn.execute(
            "SELECT * FROM images WHERE blob_id=? ORDER BY img_idx", (blob_id,)
        ).fetchall()
        return {"name": row["name"], "added_at": row["added_at"],
                "short_id": row["short_id"],
                "images": [dict(i) for i in imgs]}

    def resolve_blob_id(self, ref: str) -> str | None:
        """Return blob_id for a given short UUID or wafer name, or None if not found."""
        row = self._conn.execute(
            "SELECT blob_id FROM blobs WHERE short_id=? OR blob_id=?", (ref, ref)
        ).fetchone()
        return row["blob_id"] if row else None

    # ── FAISS fast candidate retrieval ────────────────────────────────────────

    def fast_candidates(
        self,
        query_descs_by_angle: dict,   # {angle: np.ndarray [N, 128]}
        top_k: int = FAISS_CANDIDATES,
    ) -> list[str]:
        """
        First-stage retrieval via descriptor voting.

        For each rotation, run FAISS to find the 5 nearest stored descriptors
        per query descriptor.  Each hit votes for its blob_id weighted by
        (1 - L2_dist / 2).  Returns the top_k blobs by total vote score.

        Falls back to all blobs when the DB is too small to benefit or when
        the FAISS index is empty.
        """
        all_blobs = self.blob_ids()
        if len(all_blobs) <= top_k or self._faiss_index.ntotal == 0:
            return all_blobs

        votes: dict[str, float] = {}
        for descs in query_descs_by_angle.values():
            d = descs.astype(np.float32)
            norms = np.linalg.norm(d, axis=1, keepdims=True)
            d_norm = d / (norms + 1e-6)
            D, I = self._faiss_index.search(d_norm, 5)
            for i in range(len(d_norm)):
                for j in range(5):
                    idx = int(I[i, j])
                    if idx < 0 or idx >= len(self._faiss_map):
                        continue
                    blob_idx = int(self._faiss_map[idx])
                    if blob_idx >= len(self._faiss_blob_ids):
                        continue
                    blob_id = self._faiss_blob_ids[blob_idx]
                    votes[blob_id] = votes.get(blob_id, 0.0) \
                                     + max(0.0, 1.0 - D[i, j] / 2.0)

        return sorted(all_blobs, key=lambda b: votes.get(b, 0.0), reverse=True)[:top_k]

    # ── write API ─────────────────────────────────────────────────────────────

    def add_image(
        self,
        blob_id: str,
        source_path: str,
        rgb: np.ndarray,
        gray: np.ndarray,
        disk_matcher,
    ) -> int:
        n = self._conn.execute(
            "SELECT COUNT(*) FROM images WHERE blob_id=?", (blob_id,)
        ).fetchone()[0]

        if n == 0:
            short_id = self._unique_short_id(self._conn)
            self._conn.execute(
                "INSERT OR IGNORE INTO blobs (blob_id, name, added_at, short_id)"
                " VALUES (?,?,?,?)",
                (blob_id, blob_id, _now_iso(), short_id),
            )
            bd = self.db_dir / blob_id
            bd.mkdir(parents=True, exist_ok=True)
            thumb_path = str(bd / "thumb.jpg")
            cv2.imwrite(thumb_path, cv2.cvtColor(_make_thumb(rgb, THUMB_SIZE),
                                                  cv2.COLOR_RGB2BGR))
        else:
            thumb_path = None

        disk_feats = disk_matcher.extract(gray)

        self._h5_save(blob_id, n, disk_feats)

        if not self._batch_mode:
            self._rebuild_faiss()

        self._conn.execute(
            "INSERT INTO images (blob_id, img_idx, source_path, kp_count, thumb_path)"
            " VALUES (?,?,?,?,?)",
            (blob_id, n, str(source_path),
             int(len(disk_feats["keypoints"])), thumb_path),
        )
        self._conn.commit()
        return n

    def remove_blob(self, blob_id: str) -> bool:
        if not self._conn.execute(
            "SELECT 1 FROM blobs WHERE blob_id=?", (blob_id,)
        ).fetchone():
            return False
        if self._h5_path.exists():
            with h5py.File(str(self._h5_path), "a") as f:
                key = f"blobs/{blob_id}"
                if key in f:
                    del f[key]
        shutil.rmtree(self.db_dir / blob_id, ignore_errors=True)
        self._conn.execute("DELETE FROM images WHERE blob_id=?", (blob_id,))
        self._conn.execute("DELETE FROM blobs  WHERE blob_id=?", (blob_id,))
        self._conn.commit()
        self._rebuild_faiss()
        return True

    def remove_image(self, blob_id: str, img_idx: int) -> bool:
        """Remove one image from a blob by its index.  Returns True if found.
        If it was the last image, the blob itself is also removed."""
        if not self._conn.execute(
            "SELECT 1 FROM images WHERE blob_id=? AND img_idx=?", (blob_id, img_idx)
        ).fetchone():
            return False

        # Remove from HDF5
        if self._h5_path.exists():
            with h5py.File(str(self._h5_path), "a") as f:
                key = self._h5_key(blob_id, img_idx)
                if key in f:
                    del f[key]

        self._conn.execute(
            "DELETE FROM images WHERE blob_id=? AND img_idx=?", (blob_id, img_idx)
        )

        # Remove whole blob entry if no images remain
        remaining = self._conn.execute(
            "SELECT COUNT(*) FROM images WHERE blob_id=?", (blob_id,)
        ).fetchone()[0]
        if remaining == 0:
            shutil.rmtree(self.db_dir / blob_id, ignore_errors=True)
            self._conn.execute("DELETE FROM blobs WHERE blob_id=?", (blob_id,))

        self._conn.commit()
        self._rebuild_faiss()
        return True

    def clear(self) -> None:
        self._conn.execute("DELETE FROM images")
        self._conn.execute("DELETE FROM blobs")
        self._conn.commit()
        for p in (self._h5_path, self._faiss_path, self._fmap_path, self._fids_path):
            if p.exists():
                p.unlink()
        for item in self.db_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
        self._faiss_index    = faiss.IndexScalarQuantizer(
            128, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2
        )
        self._faiss_map      = np.array([], dtype=np.int32)
        self._faiss_blob_ids = []

    # ── bulk feature loading ──────────────────────────────────────────────────

    def load_blob_features(self, blob_ids: list[str]) -> dict:
        """Load HDF5 features for the given blob_ids only."""
        result: dict[str, list] = {}
        for blob_id in blob_ids:
            blob = self.get_blob(blob_id)
            if blob is None:
                continue
            result[blob_id] = []
            for img in blob["images"]:
                try:
                    result[blob_id].append(self._h5_load(blob_id, img["img_idx"]))
                except Exception:
                    pass
        return result

    def load_all_features(self) -> dict:
        return self.load_blob_features(self.blob_ids())

    # ── migrations ────────────────────────────────────────────────────────────

    def _migrate_strip_loftr(self) -> None:
        """Remove loftr_gray datasets from existing HDF5 data and compact the file."""
        if not self._h5_path.exists():
            return
        with h5py.File(str(self._h5_path), "r") as f:
            has_loftr = any(
                "loftr_gray" in f[f"blobs/{bid}/{ikey}"]
                for bid in f.get("blobs", {})
                for ikey in f["blobs"][bid]
            )
        if not has_loftr:
            return
        print("[*] Migrating DB: removing LoFTR data from features.h5 ...")
        tmp = self._h5_path.with_suffix(".h5.tmp")
        with h5py.File(str(self._h5_path), "r") as src, \
             h5py.File(str(tmp), "w") as dst:
            for blob_id in src.get("blobs", {}).keys():
                for img_key in src["blobs"][blob_id].keys():
                    src_g = src[f"blobs/{blob_id}/{img_key}"]
                    dst_g = dst.require_group(f"blobs/{blob_id}/{img_key}")
                    for key in src_g.keys():
                        if key != "loftr_gray":
                            src.copy(f"blobs/{blob_id}/{img_key}/{key}", dst_g, name=key)
                    for k, v in src_g.attrs.items():
                        dst_g.attrs[k] = v
        tmp.replace(self._h5_path)
        print("[+] LoFTR data removed from features.h5")

    def _migrate_to_float16(self) -> None:
        """Convert disk_desc datasets from float32 to float16 (one-shot)."""
        if not self._h5_path.exists():
            return
        with h5py.File(str(self._h5_path), "r") as f:
            needs = any(
                "disk_desc" in f["blobs"][bid][ikey]
                and f["blobs"][bid][ikey]["disk_desc"].dtype == np.float32
                for bid in f.get("blobs", {})
                for ikey in f["blobs"][bid]
            )
        if not needs:
            return
        print("[*] Migrating DB: converting disk_desc to float16 ...")
        tmp = self._h5_path.with_suffix(".h5.tmp")
        with h5py.File(str(self._h5_path), "r") as src, \
             h5py.File(str(tmp), "w") as dst:
            for blob_id in src.get("blobs", {}).keys():
                for img_key in src["blobs"][blob_id].keys():
                    src_g = src[f"blobs/{blob_id}/{img_key}"]
                    dst_g = dst.require_group(f"blobs/{blob_id}/{img_key}")
                    for key in src_g.keys():
                        if key == "disk_desc":
                            dst_g.create_dataset(
                                "disk_desc",
                                data=src_g["disk_desc"][:].astype(np.float16),
                            )
                        else:
                            src.copy(f"blobs/{blob_id}/{img_key}/{key}", dst_g, name=key)
                    for k, v in src_g.attrs.items():
                        dst_g.attrs[k] = v
        tmp.replace(self._h5_path)
        print("[+] disk_desc converted to float16")

    def _migrate_legacy(self, legacy_index: Path) -> None:
        print("[*] Migrating legacy folder-based DB → SQLite + HDF5 + FAISS ...")
        with open(legacy_index) as f:
            old = json.load(f)

        for blob_id, blob in old.get("blobs", {}).items():
            self._conn.execute(
                "INSERT OR IGNORE INTO blobs (blob_id, name, added_at) VALUES (?,?,?)",
                (blob_id, blob.get("name", blob_id),
                 blob.get("added_at", _now_iso())),
            )
            for img in blob.get("images", []):
                dp = img.get("disk_path", "")
                if not (dp and Path(dp).exists()):
                    continue
                disk_data = np.load(dp)
                n = self._conn.execute(
                    "SELECT COUNT(*) FROM images WHERE blob_id=?", (blob_id,)
                ).fetchone()[0]

                self._h5_save(blob_id, n,
                              {"keypoints":   disk_data["keypoints"],
                               "descriptors": disk_data["descriptors"],
                               "scores":      disk_data["scores"],
                               "hw":          tuple(int(x) for x in disk_data["hw"])})

                self._conn.execute(
                    "INSERT INTO images (blob_id, img_idx, source_path, kp_count, thumb_path)"
                    " VALUES (?,?,?,?,?)",
                    (blob_id, n, img.get("source", ""),
                     img.get("kp_count", 0), img.get("thumb_path")),
                )

        self._conn.commit()
        self._rebuild_faiss()

        for p in self.db_dir.rglob("*_loftr.npy"):
            p.unlink(missing_ok=True)
        for p in self.db_dir.rglob("*_disk.npz"):
            p.unlink(missing_ok=True)

        legacy_index.rename(legacy_index.with_suffix(".json.bak"))
        print("[+] Migration done. Old index saved as index.json.bak")


# ── utils ─────────────────────────────────────────────────────────────────────

def _make_thumb(rgb: np.ndarray, size: int) -> np.ndarray:
    H, W = rgb.shape[:2]
    s = size / max(H, W)
    return cv2.resize(rgb, (max(1, int(W * s)), max(1, int(H * s))),
                      interpolation=cv2.INTER_AREA)
