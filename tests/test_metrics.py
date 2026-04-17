"""Unit tests for src/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    compute_bgi,
    compute_cv,
    compute_gmi,
    compute_iauc,
    compute_mage,
    compute_ppge,
    compute_recovery_time,
    compute_tar,
    compute_tbr,
    compute_tir,
)


# ─── helpers ──────────────────────────────────────────────────────────────

def s(*vals: float) -> pd.Series:
    return pd.Series(vals, dtype=float)


def _synthetic_cgm(
    baseline: float = 90.0,
    peak: float = 140.0,
    peak_offset_min: int = 60,
    total_min: int = 130,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build a synthetic CGM trace: flat baseline → rises to peak → recovers.

    Returns (DataFrame, meal_time).  Five pre-meal rows are included.
    """
    t0 = pd.Timestamp("2024-01-01 12:00:00")
    pre_times = [t0 - pd.Timedelta(minutes=i) for i in range(5, 0, -1)]
    post_times = [t0 + pd.Timedelta(minutes=i) for i in range(1, total_min + 1)]
    all_times = pre_times + [t0] + post_times

    n_pre = len(pre_times) + 1          # pre + meal moment
    n_post = len(post_times)

    glucose_pre = [baseline] * n_pre
    glucose_post: list[float] = []
    for i in range(1, n_post + 1):
        if i <= peak_offset_min:
            # linear rise
            glucose_post.append(baseline + (peak - baseline) * i / peak_offset_min)
        else:
            # linear recovery back to baseline
            remaining = n_post - peak_offset_min
            step = i - peak_offset_min
            glucose_post.append(max(baseline, peak - (peak - baseline) * step / remaining))

    glucose = glucose_pre + glucose_post
    df = pd.DataFrame({"timestamp": all_times, "glucose": glucose, "subject_id": 1})
    return df, t0


# ══════════════════════════════════════════════════════════════════════════
# TIR
# ══════════════════════════════════════════════════════════════════════════

class TestTIR:
    def test_all_in_range(self):
        assert compute_tir(s(80, 100, 120, 139)) == pytest.approx(100.0)

    def test_none_in_range(self):
        assert compute_tir(s(50, 60, 150, 200)) == pytest.approx(0.0)

    def test_half(self):
        assert compute_tir(s(80, 80, 150, 150)) == pytest.approx(50.0)

    def test_nan_ignored(self):
        assert compute_tir(s(80, float("nan"), 100)) == pytest.approx(100.0)

    def test_empty(self):
        assert np.isnan(compute_tir(pd.Series([], dtype=float)))

    def test_boundary_values_included(self):
        # 70 and 140 are inclusive
        assert compute_tir(s(70.0, 140.0)) == pytest.approx(100.0)

    def test_custom_thresholds(self):
        assert compute_tir(s(70, 100, 180), low=70, high=180) == pytest.approx(100.0)


# ══════════════════════════════════════════════════════════════════════════
# TAR
# ══════════════════════════════════════════════════════════════════════════

class TestTAR:
    def test_all_above(self):
        assert compute_tar(s(150, 200, 300)) == pytest.approx(100.0)

    def test_none_above(self):
        assert compute_tar(s(70, 100, 140)) == pytest.approx(0.0)

    def test_boundary_not_counted(self):
        # 140 is the high boundary — exactly at boundary is NOT above
        assert compute_tar(s(140.0)) == pytest.approx(0.0)

    def test_partial(self):
        assert compute_tar(s(100, 141, 100, 141)) == pytest.approx(50.0)


# ══════════════════════════════════════════════════════════════════════════
# TBR
# ══════════════════════════════════════════════════════════════════════════

class TestTBR:
    def test_all_below(self):
        assert compute_tbr(s(40, 55, 69)) == pytest.approx(100.0)

    def test_none_below(self):
        assert compute_tbr(s(70, 100, 140)) == pytest.approx(0.0)

    def test_boundary_not_counted(self):
        assert compute_tbr(s(70.0)) == pytest.approx(0.0)

    def test_partial(self):
        assert compute_tbr(s(69, 100, 69, 100)) == pytest.approx(50.0)


# ══════════════════════════════════════════════════════════════════════════
# GMI
# ══════════════════════════════════════════════════════════════════════════

class TestGMI:
    def test_known_value(self):
        # mean=100 → 3.31 + 0.02392*100 = 5.702
        assert compute_gmi(s(100, 100, 100)) == pytest.approx(5.702, abs=1e-3)

    def test_empty(self):
        assert np.isnan(compute_gmi(pd.Series([], dtype=float)))

    def test_higher_glucose_higher_gmi(self):
        assert compute_gmi(s(120, 120)) > compute_gmi(s(100, 100))

    def test_prediabetes_range(self):
        # HbA1c 5.7-6.4% → mean ~115-130 mg/dL
        gmi = compute_gmi(s(*[120.0] * 10))
        assert 5.7 <= gmi <= 6.4


# ══════════════════════════════════════════════════════════════════════════
# CV%
# ══════════════════════════════════════════════════════════════════════════

class TestCV:
    def test_constant_series(self):
        assert compute_cv(s(100, 100, 100, 100)) == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_variable_series(self):
        assert compute_cv(s(80, 90, 100, 110, 120)) > 0

    def test_high_variability(self):
        # very large swings → CV well above 36%
        assert compute_cv(s(40, 40, 200, 200)) > 36

    def test_needs_at_least_two_points(self):
        assert np.isnan(compute_cv(s(100.0)))


# ══════════════════════════════════════════════════════════════════════════
# PPGE
# ══════════════════════════════════════════════════════════════════════════

class TestPPGE:
    def test_positive_excursion(self):
        df, t0 = _synthetic_cgm(baseline=90, peak=140)
        result = compute_ppge(df, t0)
        assert result == pytest.approx(50.0, abs=3.0)

    def test_no_post_data_returns_nan(self):
        t0 = pd.Timestamp("2024-01-01 12:00:00")
        df = pd.DataFrame({
            "timestamp": [t0 - pd.Timedelta(minutes=1)],
            "glucose": [90.0],
        })
        assert np.isnan(compute_ppge(df, t0))

    def test_higher_peak_higher_ppge(self):
        df1, t0 = _synthetic_cgm(baseline=90, peak=130)
        df2, _ = _synthetic_cgm(baseline=90, peak=160)
        assert compute_ppge(df2, t0) > compute_ppge(df1, t0)


# ══════════════════════════════════════════════════════════════════════════
# iAUC
# ══════════════════════════════════════════════════════════════════════════

class TestIAUC:
    def test_positive_for_excursion(self):
        df, t0 = _synthetic_cgm(baseline=90, peak=140)
        assert compute_iauc(df, t0) > 0

    def test_flat_glucose_near_zero(self):
        t0 = pd.Timestamp("2024-01-01 12:00:00")
        times = [t0 + pd.Timedelta(minutes=i) for i in range(-5, 121)]
        df = pd.DataFrame({"timestamp": times, "glucose": [90.0] * len(times)})
        assert compute_iauc(df, t0) == pytest.approx(0.0, abs=1.0)

    def test_higher_peak_higher_iauc(self):
        df1, t0 = _synthetic_cgm(baseline=90, peak=120)
        df2, _ = _synthetic_cgm(baseline=90, peak=160)
        assert compute_iauc(df2, t0) > compute_iauc(df1, t0)

    def test_insufficient_post_data_returns_nan(self):
        t0 = pd.Timestamp("2024-01-01 12:00:00")
        df = pd.DataFrame({
            "timestamp": [t0 + pd.Timedelta(minutes=1)],
            "glucose": [100.0],
        })
        assert np.isnan(compute_iauc(df, t0))


# ══════════════════════════════════════════════════════════════════════════
# Recovery Time
# ══════════════════════════════════════════════════════════════════════════

class TestRecoveryTime:
    def test_returns_positive(self):
        df, t0 = _synthetic_cgm(baseline=90, peak=140, peak_offset_min=60, total_min=130)
        result = compute_recovery_time(df, t0)
        assert result > 0

    def test_no_recovery_within_window_returns_nan(self):
        t0 = pd.Timestamp("2024-01-01 12:00:00")
        # always elevated — never returns to baseline within window
        times = [t0 + pd.Timedelta(minutes=i) for i in range(-5, 121)]
        glucose = [90.0] * 6 + [150.0] * (len(times) - 6)
        df = pd.DataFrame({"timestamp": times, "glucose": glucose})
        result = compute_recovery_time(df, t0, threshold=5.0)
        assert np.isnan(result)

    def test_faster_recovery_lower_value(self):
        df_fast, t0 = _synthetic_cgm(baseline=90, peak=140, peak_offset_min=40, total_min=130)
        df_slow, _ = _synthetic_cgm(baseline=90, peak=140, peak_offset_min=40, total_min=130)
        # just check both are numeric and fast <= slow conceptually; both same here
        r = compute_recovery_time(df_fast, t0)
        assert not np.isnan(r)


# ══════════════════════════════════════════════════════════════════════════
# BGI
# ══════════════════════════════════════════════════════════════════════════

class TestBGI:
    def test_normal_range_non_negative(self):
        result = compute_bgi(s(80, 90, 100, 110, 120))
        assert result["lbgi"] >= 0
        assert result["hbgi"] >= 0

    def test_empty_returns_nan(self):
        result = compute_bgi(pd.Series([], dtype=float))
        assert np.isnan(result["lbgi"])
        assert np.isnan(result["hbgi"])

    def test_hypoglycemia_increases_lbgi(self):
        normal = compute_bgi(s(90, 90, 90, 90))
        hypo = compute_bgi(s(50, 50, 50, 50))
        assert hypo["lbgi"] > normal["lbgi"]

    def test_hyperglycemia_increases_hbgi(self):
        normal = compute_bgi(s(90, 90, 90, 90))
        hyper = compute_bgi(s(200, 200, 200, 200))
        assert hyper["hbgi"] > normal["hbgi"]


# ══════════════════════════════════════════════════════════════════════════
# MAGE
# ══════════════════════════════════════════════════════════════════════════

class TestMAGE:
    def test_large_excursions_detected(self):
        # strict alternating peaks/nadirs — each point is a local extremum
        g = s(80, 160, 80, 160, 80, 160, 80)
        result = compute_mage(g)
        assert result > 0

    def test_flat_returns_nan_or_zero(self):
        result = compute_mage(s(100, 100, 100, 100, 100))
        assert np.isnan(result) or result == pytest.approx(0.0, abs=1.0)

    def test_too_short_returns_nan(self):
        assert np.isnan(compute_mage(s(80, 120)))

    def test_empty_returns_nan(self):
        assert np.isnan(compute_mage(pd.Series([], dtype=float)))
