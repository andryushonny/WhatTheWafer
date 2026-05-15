"""
listener.py — background hotkey daemon for WhatTheWafer

Hotkey: Ctrl+Shift+F9
- Linux: reads /dev/input via evdev (works on Wayland and X11)
- Windows: uses pynput (no extra setup needed)

Called via:  whattw daemon

Workflow:
  1. Take a screenshot with your OS tool (PrtSc / Win+Shift+S / region select)
  2. Press Ctrl+Shift+F9 — daemon reads clipboard, identifies, shows popup with
     the query crop and the database thumbnail side by side.

Linux: if you see a permission error, fix it with one of:
  Option A — run with sudo this time:    sudo whattw daemon
  Option B — permanent fix (no re-login needed after):
             sudo usermod -aG input $USER   (then log out and back in)
"""

import os
import signal
import sys
import tempfile
import threading

import cv2
import numpy as np
import tkinter as tk

from database import BlobDB
from preprocessing import preprocess

HOTKEY_DISPLAY  = "Ctrl+Shift+F9"
_HOTKEY_PYNPUT  = "<ctrl>+<shift>+<f9>"
_HOTKEY_EVDEV   = "KEY_F9"          # evdev key code name

_MARGIN     = 20     # px from screen edge
_THUMB_SIZE = 180    # px — each image square in the popup


# ── image acquisition (clipboard) ────────────────────────────────────────────

def _get_image() -> np.ndarray:
    """Read the clipboard image as RGB uint8.

    Linux Wayland : wl-paste
    Linux X11     : xclip
    Windows       : Pillow ImageGrab
    """
    if sys.platform.startswith("linux"):
        return _get_image_linux()
    return _get_image_windows()


def _get_image_linux() -> np.ndarray:
    import subprocess

    for mime in ("image/png", "image/bmp", "image/jpeg", "image/tiff"):
        for cmd in (
            ["wl-paste", "--no-newline", "--type", mime],
            ["xclip", "-selection", "clipboard", "-t", mime, "-o"],
        ):
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=5)
                if r.returncode == 0 and r.stdout:
                    arr = np.frombuffer(r.stdout, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except FileNotFoundError:
                continue

    raise RuntimeError(
        "No image in clipboard.\n"
        "    Take a screenshot first (PrtSc / region select), then press the hotkey."
    )


def _get_image_windows() -> np.ndarray:
    from PIL import Image, ImageGrab
    img = ImageGrab.grabclipboard()
    if not isinstance(img, Image.Image):
        raise RuntimeError(
            "No image in clipboard.\n"
            "    Take a screenshot first (PrtSc / Win+Shift+S), then press the hotkey."
        )
    return np.array(img.convert("RGB"))


# ── rotation ─────────────────────────────────────────────────────────────────

def _rotate_rgb(rgb: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an RGB image by angle degrees (positive = CCW)."""
    if angle == 0:
        return rgb
    if angle % 90 == 0:
        return np.rot90(rgb, int(angle) // 90).copy()
    H, W = rgb.shape[:2]
    sin_a = abs(np.sin(np.deg2rad(angle)))
    cos_a = abs(np.cos(np.deg2rad(angle)))
    nW = int(H * sin_a + W * cos_a)
    nH = int(H * cos_a + W * sin_a)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), -angle, 1.0)
    M[0, 2] += (nW - W) / 2
    M[1, 2] += (nH - H) / 2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(
        cv2.warpAffine(bgr, M, (nW, nH), flags=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB,
    )


# ── identification ────────────────────────────────────────────────────────────

def _identify(
    rgb: np.ndarray, db: BlobDB, matcher, debug: bool = False
) -> tuple[str, int, np.ndarray | None, np.ndarray] | None:
    """Preprocess RGB, run matching.

    Returns (blob_id, inliers, ref_rgb, query_rgb_crop) or None.
    ref_rgb  — RGB image from the database to show in popup (may be None).
    query_rgb_crop — preprocessed crop of the query image.
    """
    from wafer_id import run_matching

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        proc = preprocess(tmp)
        query_crop = proc["rgb"]

        if debug:
            import time
            ts = int(time.time())
            tmp_dir = tempfile.gettempdir()
            crop_path = os.path.join(tmp_dir, f"wtw_crop_{ts}.png")
            cv2.imwrite(crop_path, cv2.cvtColor(query_crop, cv2.COLOR_RGB2BGR))
            mask_path = os.path.join(tmp_dir, f"wtw_mask_{ts}.png")
            cv2.imwrite(mask_path, proc["mask"])
            g = proc["gray"]
            print(f"  [dbg] crop → {crop_path}  ({g.shape[1]}×{g.shape[0]})")
            print(f"  [dbg] mask → {mask_path}")

        scores = run_matching(proc["gray"], db, matcher)
        if not scores:
            return None

        blob_id, inliers, best_angle = scores[0]
        query_crop = _rotate_rgb(query_crop, best_angle)
        ref_rgb = _load_ref_rgb(db, blob_id)
        return blob_id, inliers, ref_rgb, query_crop
    finally:
        os.unlink(tmp)


def _load_ref_rgb(db: BlobDB, blob_id: str) -> np.ndarray | None:
    """Return an RGB image for the matched blob.

    Tries stored thumbnail first; falls back to the source image file.
    Returns None only if nothing is readable.
    """
    blob = db.get_blob(blob_id)
    if not blob:
        return None

    for img_row in blob["images"]:
        # 1) stored thumbnail (JPEG crop with keypoints)
        tp = img_row.get("thumb_path")
        if tp and os.path.exists(tp):
            img = cv2.imread(tp, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2) original source image
        sp = img_row.get("source_path")
        if sp and os.path.exists(sp):
            try:
                return preprocess(sp)["rgb"]   # returns blob crop, same as query
            except Exception:
                pass

    return None


# ── popup helpers ─────────────────────────────────────────────────────────────

def _rgb_to_photoimage(rgb: np.ndarray, size: int) -> tk.PhotoImage:
    """Resize an RGB numpy array to a square and return a tkinter PhotoImage."""
    h, w = rgb.shape[:2]
    # Square crop from centre
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    sq = rgb[y0:y0 + side, x0:x0 + side]
    sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    cv2.imwrite(tmp, cv2.cvtColor(sq, cv2.COLOR_RGB2BGR))
    photo = tk.PhotoImage(file=tmp)
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return photo


def _load_thumb_photoimage(thumb_path: str, size: int) -> tk.PhotoImage | None:
    """Load a JPEG thumbnail from disk and return a tkinter PhotoImage."""
    try:
        img = cv2.imread(thumb_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return _rgb_to_photoimage(rgb, size)
    except Exception:
        return None


# ── popup ─────────────────────────────────────────────────────────────────────

def _show_popup(
    root: tk.Tk,
    result: tuple[str, int, np.ndarray | None, np.ndarray] | None,
    threshold: int,
) -> tk.Toplevel:
    """Render an always-on-top result popup with query + DB thumbnail side by side."""
    popup = tk.Toplevel(root)
    popup.attributes("-topmost", True)
    popup.attributes("-alpha", 0.93)
    popup.overrideredirect(True)

    # ── colours ──────────────────────────────────────────────────────────────
    if result is None:
        title, bg, fg = "[?]  No match", "#1e1e1e", "#ff9944"
        ref_rgb, query_crop = None, None
    else:
        blob_id, inliers, ref_rgb, query_crop = result
        if inliers >= threshold:
            title = f"[+]  {blob_id}   ({inliers} inliers)"
            bg, fg = "#0f1f0f", "#66ff88"
        else:
            title = f"[?]  {blob_id}   ({inliers} < {threshold})"
            bg, fg = "#1f160a", "#ffbb44"

    popup.configure(bg=bg)

    # ── title row ─────────────────────────────────────────────────────────────
    tk.Label(
        popup, text=title,
        font=("monospace", 22, "bold"),
        bg=bg, fg=fg,
        padx=24, pady=12,
    ).pack(fill="x")

    # ── image row (only when there's a match) ─────────────────────────────────
    if query_crop is not None:
        img_frame = tk.Frame(popup, bg=bg)
        img_frame.pack(padx=24, pady=(0, 20))

        # Query image (left)
        try:
            q_photo = _rgb_to_photoimage(query_crop, _THUMB_SIZE)
            q_col = tk.Frame(img_frame, bg=bg)
            q_col.pack(side="left", padx=(0, 16))
            lbl_q = tk.Label(q_col, image=q_photo, bg=bg)
            lbl_q.image = q_photo   # keep reference
            lbl_q.pack()
            tk.Label(q_col, text="Query", font=("monospace", 11),
                     bg=bg, fg="#888888").pack()
        except Exception:
            pass

        # DB thumbnail (right)
        if ref_rgb is not None:
            db_photo = _rgb_to_photoimage(ref_rgb, _THUMB_SIZE)
            if db_photo:
                d_col = tk.Frame(img_frame, bg=bg)
                d_col.pack(side="left")
                lbl_d = tk.Label(d_col, image=db_photo, bg=bg)
                lbl_d.image = db_photo
                lbl_d.pack()
                tk.Label(d_col, text="Database", font=("monospace", 11),
                         bg=bg, fg="#888888").pack()

    # ── position top-right ────────────────────────────────────────────────────
    popup.update_idletasks()
    sw = popup.winfo_screenwidth()
    w  = popup.winfo_reqwidth()
    popup.geometry(f"+{sw - w - _MARGIN}+{_MARGIN}")

    popup.bind("<Escape>",    lambda _e: popup.destroy())
    popup.bind("<Button-1>",  lambda _e: popup.destroy())
    popup.focus_force()
    return popup


# ── platform-specific hotkey listeners ───────────────────────────────────────

def _start_linux_listener(callback) -> None:
    """Listen for the hotkey via evdev. Works on Wayland and X11."""
    import select
    import evdev
    from evdev import ecodes

    CTRL    = {ecodes.KEY_LEFTCTRL,  ecodes.KEY_RIGHTCTRL}
    SHIFT   = {ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT}
    TRIGGER = getattr(ecodes, _HOTKEY_EVDEV)

    def _is_hotkey(pressed: set) -> bool:
        return bool(CTRL & pressed) and bool(SHIFT & pressed) and TRIGGER in pressed

    def _run() -> None:
        keyboards = []
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                if ecodes.KEY_A in dev.capabilities().get(ecodes.EV_KEY, []):
                    keyboards.append(dev)
            except (PermissionError, OSError):
                pass

        if not keyboards:
            _handle_no_devices()
            return

        pressed: set[int] = set()
        firing = False

        while keyboards:
            r, _, _ = select.select(keyboards, [], [], 1.0)
            for dev in r:
                try:
                    for ev in dev.read():
                        if ev.type != ecodes.EV_KEY:
                            continue
                        if ev.value == 1:       # key down
                            pressed.add(ev.code)
                        elif ev.value == 0:     # key up
                            pressed.discard(ev.code)
                            if TRIGGER not in pressed:
                                firing = False
                        if not firing and _is_hotkey(pressed):
                            firing = True
                            callback()
                except OSError:
                    keyboards = [k for k in keyboards if k is not dev]

    threading.Thread(target=_run, daemon=True).start()


def _handle_no_devices() -> None:
    can_open = False
    for path in [f"/dev/input/event{i}" for i in range(8)]:
        try:
            open(path, "rb").close()
            can_open = True
            break
        except PermissionError:
            break
        except FileNotFoundError:
            continue

    if not can_open:
        print("[!] Cannot read /dev/input — no permission.")
        print()
        print("    Fix option A (this session only):")
        print("      sudo whattw daemon")
        print()
        print("    Fix option B (permanent, needs one re-login):")
        print("      sudo usermod -aG input $USER")
        print("      # then log out and back in")
    else:
        print("[!] No keyboard found in /dev/input.")
    os._exit(1)


def _start_windows_listener(callback) -> None:
    from pynput import keyboard as pk

    def _run() -> None:
        with pk.GlobalHotKeys({_HOTKEY_PYNPUT: callback}) as h:
            h.join()

    threading.Thread(target=_run, daemon=True).start()


# ── daemon entry point ────────────────────────────────────────────────────────

def cmd_daemon(args) -> None:
    root = tk.Tk()
    root.withdraw()

    state: dict = {"matcher": None, "db": None, "busy": False, "popup": None}
    lock = threading.Lock()
    debug: bool = getattr(args, "debug", False)

    def worker() -> None:
        with lock:
            if state["busy"]:
                return
            state["busy"] = True
        try:
            if state["matcher"] is None:
                print("[*] First run — loading DISK+LightGlue (~10 s)...")
                from wafer_id import _load_matcher, _device
                device = _device(getattr(args, "no_gpu", False))
                state["matcher"] = _load_matcher(device, verbose=False)
                state["db"] = BlobDB(args.db)
                print("[*] Models ready.\n")

            rgb = _get_image()

            if debug:
                import time
                ts = int(time.time())
                raw_path = os.path.join(tempfile.gettempdir(), f"wtw_raw_{ts}.png")
                cv2.imwrite(raw_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                print(f"  [dbg] clipboard → {raw_path}  ({rgb.shape[1]}×{rgb.shape[0]})")

            result = _identify(rgb, state["db"], state["matcher"], debug=debug)

            def show(r=result):
                p = state["popup"]
                if p is not None:
                    try:
                        p.destroy()
                    except Exception:
                        pass
                state["popup"] = _show_popup(root, r, args.threshold)

            root.after(0, show)

            if result:
                blob_id, inliers, _, _ = result
                mark = "[+]" if inliers >= args.threshold else "[?]"
                print(f"  {mark}  {blob_id}  ({inliers} inliers)")
            else:
                print("  ?  No match")
        except Exception as exc:
            print(f"[!] {exc}")
        finally:
            with lock:
                state["busy"] = False

    def on_hotkey() -> None:
        threading.Thread(target=worker, daemon=True).start()

    if sys.platform.startswith("linux"):
        _start_linux_listener(on_hotkey)
    else:
        _start_windows_listener(on_hotkey)

    def _stop(*_):
        print("\n[*] Stopped.")
        os._exit(0)

    signal.signal(signal.SIGINT, _stop)

    print(f"[*] WhatTheWafer daemon running — hotkey: {HOTKEY_DISPLAY}")
    print(f"    Database : {args.db}")
    print(f"    Threshold: {args.threshold} inliers")
    print(f"    Workflow : screenshot → clipboard → {HOTKEY_DISPLAY} → popup")
    print(f"    Press Ctrl+C to stop.\n")

    root.mainloop()
