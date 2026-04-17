"""Blood glucose metrics library.

Basic 5  (industry standard):  TIR · TAR · TBR · GMI · CV%
Extended core 3 (differentiation): PPGE · iAUC · Recovery Time
Optional extras:  BGI · MAGE

All scalar functions accept a pd.Series of glucose values (mg/dL).
Meal-level functions additionally accept the full tidy CGM DataFrame
and a meal timestamp.
"""

import numpy as np
import pandas as pd

from .config import (
    GLUCOSE_HIGH,
    GLUCOSE_LOW,
    POSTPRANDIAL_WINDOW,
    RECOVERY_THRESHOLD,
)


# ══════════════════════════════════════════════════════════════════════════
# Basic 5 — industry standard
# ══════════════════════════════════════════════════════════════════════════

def compute_tir(
    glucose: pd.Series,
    low: float = GLUCOSE_LOW,
    high: float = GLUCOSE_HIGH,
) -> float:
    """Time In Range (%) — fraction of readings in [low, high]."""
    g = glucose.dropna()
    if len(g) == 0:
        return float("nan")
    return float(((g >= low) & (g <= high)).mean() * 100)


def compute_tar(glucose: pd.Series, high: float = GLUCOSE_HIGH) -> float:
    """Time Above Range (%) — fraction of readings > high."""
    g = glucose.dropna()
    if len(g) == 0:
        return float("nan")
    return float((g > high).mean() * 100)


def compute_tbr(glucose: pd.Series, low: float = GLUCOSE_LOW) -> float:
    """Time Below Range (%) — fraction of readings < low."""
    g = glucose.dropna()
    if len(g) == 0:
        return float("nan")
    return float((g < low).mean() * 100)


def compute_gmi(glucose: pd.Series) -> float:
    """Glucose Management Indicator (%) — HbA1c proxy from mean glucose.

    Formula: GMI = 3.31 + 0.02392 × mean_glucose  (Bergenstal 2018)
    """
    g = glucose.dropna()
    if len(g) == 0:
        return float("nan")
    return float(3.31 + 0.02392 * g.mean())


def compute_cv(glucose: pd.Series) -> float:
    """Coefficient of Variation (%) — SD / mean × 100.

    <36% is considered stable (international consensus).
    """
    g = glucose.dropna()
    if len(g) < 2 or g.mean() == 0:
        return float("nan")
    return float(g.std(ddof=1) / g.mean() * 100)


# ══════════════════════════════════════════════════════════════════════════
# Extended core 3 — meal-level, differentiation
# ══════════════════════════════════════════════════════════════════════════

def _premeal_baseline(
    cgm: pd.DataFrame,
    meal_time: pd.Timestamp,
    gl_col: str = "glucose",
    ts_col: str = "timestamp",
    lookback_min: int = 10,
) -> float:
    """Median glucose in the lookback window before meal_time."""
    start = meal_time - pd.Timedelta(minutes=lookback_min)
    window = cgm[(cgm[ts_col] >= start) & (cgm[ts_col] <= meal_time)][gl_col]
    return float(window.median()) if len(window) > 0 else float("nan")


def compute_ppge(
    cgm: pd.DataFrame,
    meal_time: pd.Timestamp,
    window_min: int = POSTPRANDIAL_WINDOW,
    gl_col: str = "glucose",
    ts_col: str = "timestamp",
) -> float:
    """Post-Prandial Glucose Excursion (mg/dL) = peak − premeal baseline.

    References: IDF guideline; postprandial 2h <140 mg/dL recommended.
    """
    baseline = _premeal_baseline(cgm, meal_time, gl_col, ts_col)
    end = meal_time + pd.Timedelta(minutes=window_min)
    post = cgm[(cgm[ts_col] > meal_time) & (cgm[ts_col] <= end)][gl_col].dropna()
    if np.isnan(baseline) or len(post) == 0:
        return float("nan")
    return float(post.max() - baseline)


def compute_iauc(
    cgm: pd.DataFrame,
    meal_time: pd.Timestamp,
    window_min: int = POSTPRANDIAL_WINDOW,
    gl_col: str = "glucose",
    ts_col: str = "timestamp",
) -> float:
    """Incremental AUC above premeal baseline (mg/dL × min), positive only.

    Uses trapezoidal integration; excursions below baseline are clamped to 0.
    References: Zeevi et al. Cell 2015; CGMacros paper (iAUC as XGBoost target).
    """
    baseline = _premeal_baseline(cgm, meal_time, gl_col, ts_col)
    end = meal_time + pd.Timedelta(minutes=window_min)
    post = (
        cgm[(cgm[ts_col] > meal_time) & (cgm[ts_col] <= end)][[ts_col, gl_col]]
        .dropna()
        .copy()
    )
    if np.isnan(baseline) or len(post) < 2:
        return float("nan")
    t = (post[ts_col] - post[ts_col].iloc[0]).dt.total_seconds() / 60.0
    excursion = (post[gl_col].values - baseline).clip(min=0)
    # 최신 버전에서는 trapezoid라는 이름을 사용합니다.
    # area = np.trapezoid(y, x)
    #return float(np.trapz(excursion, t))
    return float(np.trapezoid(excursion, t))


def compute_recovery_time(
    cgm: pd.DataFrame,
    meal_time: pd.Timestamp,
    window_min: int = POSTPRANDIAL_WINDOW,
    gl_col: str = "glucose",
    ts_col: str = "timestamp",
    threshold: float = RECOVERY_THRESHOLD,
) -> float:
    """Minutes until glucose returns within ±threshold of premeal baseline.

    Returns NaN if recovery not observed within window_min.
    <90 min: good insulin sensitivity; >150 min: possible resistance.
    """
    baseline = _premeal_baseline(cgm, meal_time, gl_col, ts_col)
    end = meal_time + pd.Timedelta(minutes=window_min)
    post = (
        cgm[(cgm[ts_col] > meal_time) & (cgm[ts_col] <= end)][[ts_col, gl_col]]
        .dropna()
    )
    if np.isnan(baseline) or len(post) == 0:
        return float("nan")
    recovered = post[np.abs(post[gl_col] - baseline) <= threshold]
    if len(recovered) == 0:
        return float("nan")
    return float((recovered[ts_col].iloc[0] - meal_time).total_seconds() / 60.0)


# ══════════════════════════════════════════════════════════════════════════
# Optional extras
# ══════════════════════════════════════════════════════════════════════════

def compute_bgi(glucose: pd.Series) -> dict[str, float]:
    """Blood Glucose Index — asymmetric risk decomposition.

    Returns {'lbgi': ..., 'hbgi': ...}.
    Reference: Kovatchev et al. Diabetes Technology & Therapeutics 2006.
    """
    g = glucose.dropna()
    if len(g) == 0:
        return {"lbgi": float("nan"), "hbgi": float("nan")}
    f = 1.509 * (np.log(g.values) ** 1.084 - 5.381)
    rl = (np.minimum(f, 0) ** 2) * 10
    rh = (np.maximum(f, 0) ** 2) * 10
    return {"lbgi": float(rl.mean()), "hbgi": float(rh.mean())}


def compute_mage(glucose: pd.Series) -> float:
    """Mean Amplitude of Glycemic Excursions (mg/dL).

    Only excursions larger than 1 SD of the series are counted.
    Reference: Service et al. Diabetes 1970.
    Note: implementation complexity is intentional — demonstrates
    algorithm skill beyond simple aggregations.
    """
    g = glucose.dropna().values
    if len(g) < 3:
        return float("nan")
    sd = g.std(ddof=1)
    if sd == 0:
        return float("nan")

    # local extrema (naive peak/nadir detection)
    extrema_idx = [
        i for i in range(1, len(g) - 1)
        if (g[i] > g[i - 1] and g[i] > g[i + 1])
        or (g[i] < g[i - 1] and g[i] < g[i + 1])
    ]
    if len(extrema_idx) < 2:
        return float("nan")

    excursions = np.abs(np.diff(g[extrema_idx]))
    significant = excursions[excursions > sd]
    return float(significant.mean()) if len(significant) > 0 else float("nan")


# ══════════════════════════════════════════════════════════════════════════
# Batch helpers
# ══════════════════════════════════════════════════════════════════════════

def compute_subject_summary(df: pd.DataFrame, gl_col: str = "glucose") -> pd.DataFrame:
    """Compute basic-5 metrics for every subject across their full recording.

    Parameters
    ----------
    df : combined CGM DataFrame with 'subject_id', 'timestamp', glucose column.

    Returns
    -------
    DataFrame indexed by subject_id with columns:
    tir, tar, tbr, gmi, cv_pct, mean_glucose, n_readings, lbgi, hbgi.
    """
    rows = []
    for sid, grp in df.groupby("subject_id"):
        g = grp[gl_col]
        bgi = compute_bgi(g)
        rows.append(
            {
                "subject_id": sid,
                "tir": compute_tir(g),
                "tar": compute_tar(g),
                "tbr": compute_tbr(g),
                "gmi": compute_gmi(g),
                "cv_pct": compute_cv(g),
                "mean_glucose": float(g.mean()),
                "n_readings": int(g.notna().sum()),
                "lbgi": bgi["lbgi"],
                "hbgi": bgi["hbgi"],
            }
        )
    return pd.DataFrame(rows).set_index("subject_id")


def compute_meal_metrics(
    df: pd.DataFrame,
    gl_col: str = "glucose",
    ts_col: str = "timestamp",
    window_min: int = POSTPRANDIAL_WINDOW,
) -> pd.DataFrame:
    """Compute PPGE, iAUC, and Recovery Time for every logged meal.

    Parameters
    ----------
    df : combined CGM DataFrame; must contain 'is_meal' column.
    """
    meal_rows = df[df["is_meal"]].copy()
    results = []
    for _, meal in meal_rows.iterrows():
        sid = meal["subject_id"]
        t0 = meal[ts_col]
        subj_df = df[df["subject_id"] == sid]
        results.append(
            {
                "subject_id": sid,
                "meal_time": t0,
                "meal_type": meal.get("meal_type"),
                "carbs": meal.get("carbs"),
                "protein": meal.get("protein"),
                "fat": meal.get("fat"),
                "fiber": meal.get("fiber"),
                "meal_calories": meal.get("meal_calories"),
                "ppge": compute_ppge(subj_df, t0, window_min, gl_col, ts_col),
                "iauc": compute_iauc(subj_df, t0, window_min, gl_col, ts_col),
                "recovery_min": compute_recovery_time(subj_df, t0, window_min, gl_col, ts_col),
            }
        )
    return pd.DataFrame(results).reset_index(drop=True)
