# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 12:07:51 2026

@author: connort2
"""

# -*- coding: utf-8 -*-
"""
Miguel VR frequency bands: Relative band power from rawHemodynamicSubtr (bilateral ROIs)

Folder pattern expected:
<SESSION_ROOT>\<ANIMAL_ID>\processed\brain\allen\rawBlue_rawViolet_rawHemodynamicSubtr\*_bilateral_rawBlue_rawViolet_rawHemodynamicSubtr.mat

Assumptions:
- Sampling raI:\data\LRI-110044\2025\April25\042225_TC\GEMM_6s_Group1_BlenderSess1I:\data\LRI-110044\2025\May25\050825_TC\GEMM_8m_Group1_Blender_Sess1_40HzI:\data\LRI-110044\2025\April25\042225_TC\GEMM_6s_Group1_BlenderSess1I:\data\LRI-110044\2025\April25\042925_TC\GEMM_Group1_6s_VRVirmen_Sess1te fs = 20 Hz (fixed)
- Compute relative power using Option A:
    rel_band = band_power / total_power_in_(0.01–9.99 Hz)
- Bands:
    delta: 0.01–4 Hz
    theta: 4–8 Hz
    alpha: 8–9.99 Hz
- Only uses rawHemodynamicSubtr variable inside each .mat

Outputs (to user-selected output folder):
- relative_power_long.csv  (tidy/long format, best for stats/GraphPad)
- relative_power_wide.csv  (wide format, convenient for heatmaps)
- file_manifest.csv        (what files were processed / skipped)
"""


import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch

# -------------------- fixed analysis settings --------------------
FS_HZ = 20.0
PSD_TOTAL_BAND = (0.01, 9.99)

BANDS_HZ = {
    "delta_0p01_4": (0.01, 4.0),
    "theta_4_8": (4.0, 8.0),
    "alpha_8_9p99": (8.0, 9.99),
}

# Welch parameters (safe defaults; change if you want)
# With fs=20, nperseg=1024 is too large for many recordings; auto is usually fine.
WELCH_NPERSEG = None     # e.g., 512 if you want fixed
WELCH_NOOVERLAP = None   # e.g., 256 if you set nperseg


# -------------------- GUI helpers --------------------
def pick_folder_gui(title: str) -> Path:
    """Tk folder picker (works well in Spyder on Windows)."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    if not folder:
        raise RuntimeError(f"No folder selected for: {title}")
    return Path(folder)


# -------------------- parsing helpers --------------------
def find_animal_folders(session_root: Path) -> list[Path]:
    """Animal folders are numeric directory names under the session root (e.g., 430)."""
    animals = [p for p in session_root.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)]
    return sorted(animals, key=lambda p: int(p.name))


def extract_roi_tokens_from_filename(mat_path: Path) -> dict:
    """
    Example filename:
      MOp1_bilateral_rawBlue_rawViolet_rawHemodynamicSubtr.mat
      SSp_tr1_bilateral_rawBlue_rawViolet_rawHemodynamicSubtr.mat

    roi_full = token before "_bilateral_"
    region_base = roi_full with trailing digits stripped (anything before the final digit(s))
    """
    stem = mat_path.stem
    m = re.match(r"^(?P<roi_full>.+?)_bilateral_", stem)
    roi_full = m.group("roi_full") if m else stem.split("_bilateral_")[0]

    m2 = re.match(r"^(?P<region_base>.+?)(?P<roi_num>\d+)$", roi_full)
    if m2:
        region_base = m2.group("region_base")
        roi_num = m2.group("roi_num")
    else:
        region_base = roi_full
        roi_num = ""

    return {"roi_full": roi_full, "region_base": region_base, "roi_num": roi_num}


def load_raw_hemodynamic_subtr(mat_path: Path) -> tuple[np.ndarray, str]:
    """
    Load the rawHemodynamicSubtr 1D signal from a .mat.

    This looks for a variable key containing 'hemodynamicSubtr' AND 'raw' (case-insensitive).
    If multiple match, it picks the first (stable sorted order).
    """
    md = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    keys = sorted([k for k in md.keys() if not k.startswith("__")])

    candidates = []
    for k in keys:
        kl = k.lower()
        if ("hemodynamicsubtr" in kl) and ("raw" in kl):
            candidates.append(k)

    if not candidates:
        raise KeyError(
            f"No rawHemodynamicSubtr-like variable found in {mat_path.name}. "
            f"Available keys: {keys}"
        )

    chosen = candidates[0]
    x = np.asarray(md[chosen]).astype(np.float32).reshape(-1)
    return x, chosen


# -------------------- PSD / bandpower --------------------
def bandpower_welch(x: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """Integrate Welch PSD over band [fmin, fmax)."""
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan

    f, pxx = welch(
        x,
        fs=fs,
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOOVERLAP,
        detrend="constant",
        scaling="density",
    )
    fmin, fmax = band
    mask = (f >= fmin) & (f < fmax)
    if not np.any(mask):
        return np.nan
    return float(np.trapz(pxx[mask], f[mask]))


def compute_band_metrics(x: np.ndarray, fs: float) -> dict[str, float]:
    """
    Computes absolute band power, total power (0.01–9.99), and relative band power.
    """
    total = bandpower_welch(x, fs, PSD_TOTAL_BAND)
    out = {
        "totalPower_0p01_9p99": total,
        "n_samples": int(np.isfinite(x).sum()),
    }

    for band_name, band in BANDS_HZ.items():
        abs_p = bandpower_welch(x, fs, band)
        rel_p = (abs_p / total) if (np.isfinite(abs_p) and np.isfinite(total) and total > 0) else np.nan
        out[f"abs_{band_name}"] = abs_p
        out[f"rel_{band_name}"] = rel_p

    return out


# -------------------- main pipeline --------------------
def main():
    session_root = pick_folder_gui("Select SESSION ROOT (e.g., ...Virmen_Sess1_40Hz)")
    out_dir = pick_folder_gui("Select OUTPUT folder (where CSVs will be saved)")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSession root: {session_root}")
    print(f"Output folder: {out_dir}")
    print(f"fs = {FS_HZ} Hz | total band = {PSD_TOTAL_BAND} Hz")
    print("Bands:", BANDS_HZ)

    manifest_rows = []
    long_rows = []
    wide_rows = []

    animal_dirs = find_animal_folders(session_root)
    if not animal_dirs:
        raise RuntimeError(f"No numeric animal folders found under: {session_root}")

    for animal_dir in animal_dirs:
        animal_id = animal_dir.name
        target = animal_dir / "processed" / "brain" / "allen" / "rawBlue_rawViolet_rawHemodynamicSubtr"

        if not target.exists():
            manifest_rows.append({
                "animal_id": animal_id,
                "status": "missing_target_folder",
                "target_folder": str(target),
                "file": "",
                "note": ""
            })
            continue

        mats = sorted(target.glob("*_bilateral_rawBlue_rawViolet_rawHemodynamicSubtr.mat"))
        if not mats:
            manifest_rows.append({
                "animal_id": animal_id,
                "status": "no_bilateral_mat_files",
                "target_folder": str(target),
                "file": "",
                "note": ""
            })
            continue

        print(f"\nAnimal {animal_id}: {len(mats)} bilateral ROI files found")

        for mat_path in mats:
            tokens = extract_roi_tokens_from_filename(mat_path)
            roi_full = tokens["roi_full"]
            region_base = tokens["region_base"]

            try:
                x, varname = load_raw_hemodynamic_subtr(mat_path)
                metrics = compute_band_metrics(x, FS_HZ)

                manifest_rows.append({
                    "animal_id": animal_id,
                    "status": "ok",
                    "target_folder": str(target),
                    "file": str(mat_path),
                    "note": f"used_var={varname}"
                })

                # Build wide row
                wide = {
                    "session_root": str(session_root),
                    "animal_id": animal_id,
                    "roi_full": roi_full,
                    "region_base": region_base,
                    "mat_file": str(mat_path),
                    "used_var": varname,
                    "fs_hz": FS_HZ,
                    **metrics
                }
                wide_rows.append(wide)

                # Build long rows (one row per band)
                for band_name in BANDS_HZ.keys():
                    long_rows.append({
                        "session_root": str(session_root),
                        "animal_id": animal_id,
                        "roi_full": roi_full,
                        "region_base": region_base,
                        "mat_file": str(mat_path),
                        "used_var": varname,
                        "fs_hz": FS_HZ,
                        "band": band_name,
                        "abs_power": metrics.get(f"abs_{band_name}", np.nan),
                        "rel_power": metrics.get(f"rel_{band_name}", np.nan),
                        "total_power_0p01_9p99": metrics.get("totalPower_0p01_9p99", np.nan),
                        "n_samples": metrics.get("n_samples", np.nan),
                    })

            except Exception as e:
                manifest_rows.append({
                    "animal_id": animal_id,
                    "status": "error",
                    "target_folder": str(target),
                    "file": str(mat_path),
                    "note": repr(e)
                })
                continue

    # Save outputs
    manifest_df = pd.DataFrame(manifest_rows)
    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)

    manifest_path = out_dir / "file_manifest.csv"
    long_path = out_dir / "relative_power_long.csv"
    wide_path = out_dir / "relative_power_wide.csv"

    manifest_df.to_csv(manifest_path, index=False)
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    print("\nSaved:")
    print(f"  {manifest_path}")
    print(f"  {long_path}")
    print(f"  {wide_path}")

    # Quick sanity summary
    if not wide_df.empty:
        print("\nQuick check (mean relative power across all rows):")
        rel_cols = [c for c in wide_df.columns if c.startswith("rel_")]
        print(wide_df[rel_cols].mean(numeric_only=True).round(4))


if __name__ == "__main__":
    main()