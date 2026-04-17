"""Functions to load and clean CGMacros dataset files."""

import re
from pathlib import Path

import pandas as pd

from .config import RAW_DATA_DIR, COL_MAP_BIO, COL_MAP_CGM


def load_bio() -> pd.DataFrame:
    """Load participant demographics from bio.csv.

    Returns a tidy DataFrame with snake_case column names.
    Only the columns defined in COL_MAP_BIO are retained; extras are dropped.
    """
    df = pd.read_csv(RAW_DATA_DIR / "bio.csv")
    rename = {k: v for k, v in COL_MAP_BIO.items() if k in df.columns}
    df = df.rename(columns=rename)[list(rename.values())]
    df["subject_id"] = df["subject_id"].astype(int)
    return df.reset_index(drop=True)


def _parse_subject_id(path: Path) -> int:
    m = re.search(r"CGMacros-(\d+)", path.name)
    return int(m.group(1)) if m else -1


def load_cgm(subject_id: int) -> pd.DataFrame:
    """Load one participant's minute-level CGM + meal file.

    Adds:
    - subject_id column
    - glucose column  (Libre GL preferred; Dexcom GL as fallback)
    - is_meal column  (True on rows where a meal was logged)
    """
    sid = f"{subject_id:03d}"
    csv_path = RAW_DATA_DIR / f"CGMacros-{sid}" / f"CGMacros-{sid}.csv"
    df = pd.read_csv(csv_path)

    rename = {k: v for k, v in COL_MAP_CGM.items() if k in df.columns}
    df = df.rename(columns=rename)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["subject_id"] = subject_id
    df["glucose"] = df["libre_gl"].fillna(df.get("dexcom_gl", pd.Series(dtype=float)))
    df["is_meal"] = df["meal_type"].notna()

    drop_cols = [c for c in ["Unnamed: 0", "image_path"] if c in df.columns]
    return df.drop(columns=drop_cols).reset_index(drop=True)


def load_all_cgm(verbose: bool = False) -> pd.DataFrame:
    """Load and concatenate every participant's CGM data.

    Parameters
    ----------
    verbose : print subject IDs as they are loaded
    """
    frames: list[pd.DataFrame] = []
    subject_dirs = sorted(
        [d for d in RAW_DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("CGMacros-")]
    )
    for d in subject_dirs:
        sid = _parse_subject_id(d)
        if sid < 0:
            continue
        try:
            frames.append(load_cgm(sid))
            if verbose:
                print(f"  loaded subject {sid:03d}")
        except FileNotFoundError:
            if verbose:
                print(f"  SKIP {sid:03d} (file not found)")

    if not frames:
        raise RuntimeError(f"No CGM data found under {RAW_DATA_DIR}")
    return pd.concat(frames, ignore_index=True)


def get_meal_events(df: pd.DataFrame) -> pd.DataFrame:
    """Extract rows where a meal was logged; one row = one meal event."""
    return (
        df[df["is_meal"]]
        .copy()
        .reset_index(drop=True)
    )
