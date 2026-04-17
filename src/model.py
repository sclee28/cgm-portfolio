"""PPGR prediction pipeline: baselines, population XGBoost, personalized LOSO."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_XGB: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "n": int(len(yt)),
    }


def _make_xgb(overrides: dict | None = None) -> xgb.XGBRegressor:
    params = {**_DEFAULT_XGB}
    if overrides:
        params.update(overrides)
    return xgb.XGBRegressor(**params)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline models
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_baselines(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> list[dict]:
    """Mean baseline and ridge regression evaluated with GroupKFold."""
    cv = GroupKFold(n_splits=n_splits)
    y_arr = y.to_numpy()

    results = []

    mean_preds = cross_val_predict(
        DummyRegressor(strategy="mean"), X, y, groups=groups, cv=cv
    )
    results.append({"model_type": "mean_baseline", **_metrics(y_arr, mean_preds),
                    "oof_preds": mean_preds})

    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    ridge_preds = cross_val_predict(ridge, X, y, groups=groups, cv=cv)
    results.append({"model_type": "ridge_regression", **_metrics(y_arr, ridge_preds),
                    "oof_preds": ridge_preds})

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Population XGBoost — GroupKFold
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_population_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    xgb_params: dict | None = None,
) -> dict:
    """XGBoost population model with GroupKFold cross-validation.

    GroupKFold guarantees no subject leaks between train and validation folds,
    which is critical: a model trained on the same subject it predicts would
    inflate R² via memorisation rather than generalisation.
    """
    model = _make_xgb(xgb_params)
    cv = GroupKFold(n_splits=n_splits)
    y_arr = y.to_numpy()

    oof = cross_val_predict(model, X, y, groups=groups, cv=cv)

    return {
        "model_type": "population_xgboost",
        "cv": f"GroupKFold(n_splits={n_splits})",
        **_metrics(y_arr, oof),
        "oof_preds": oof,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Personalized model — Leave-One-Subject-Out (LOSO)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_personalized_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    xgb_params: dict | None = None,
    min_test_meals: int = 5,
    verbose: bool = False,
) -> dict:
    """Personalized XGBoost via Leave-One-Subject-Out cross-validation.

    For each subject:
      1. Train XGBoost on all other subjects (population knowledge).
      2. Use the first 50 % of the held-out subject's meals to compute
         a subject-specific bias correction (personalisation step).
      3. Apply bias-corrected predictions on the remaining 50 % of meals.

    This simulates a real deployment scenario where a new user accumulates
    a few meals before the model adapts to their personal glucose response.
    """
    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    g_arr = groups.to_numpy()

    oof = np.full(len(y_arr), np.nan)
    unique_sids = np.unique(g_arr)

    for sid in unique_sids:
        test_mask = g_arr == sid
        train_mask = ~test_mask

        if test_mask.sum() < min_test_meals or train_mask.sum() < 20:
            if verbose:
                print(f"  skip subject {sid:03d} (too few meals)")
            continue

        model = _make_xgb(xgb_params)
        model.fit(X_arr[train_mask], y_arr[train_mask])

        test_idx = np.where(test_mask)[0]
        n_cal = max(1, len(test_idx) // 2)
        cal_idx, val_idx = test_idx[:n_cal], test_idx[n_cal:]

        # personalisation: compute bias on calibration half
        cal_preds = model.predict(X_arr[cal_idx])
        bias = float(np.mean(y_arr[cal_idx] - cal_preds))

        oof[cal_idx] = cal_preds                           # before bias correction
        if len(val_idx) > 0:
            oof[val_idx] = model.predict(X_arr[val_idx]) + bias  # after correction

        if verbose:
            print(f"  subject {sid:03d}: n_train={train_mask.sum()}, "
                  f"n_test={test_mask.sum()}, bias={bias:+.1f}")

    valid = ~np.isnan(oof)
    return {
        "model_type": "personalized_xgboost",
        "strategy": "LOSO + per-subject bias correction",
        **_metrics(y_arr[valid], oof[valid]),
        "oof_preds": oof,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Final model — trained on all data for SHAP
# ──────────────────────────────────────────────────────────────────────────────

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    xgb_params: dict | None = None,
) -> xgb.XGBRegressor:
    """Train XGBoost on the full dataset for SHAP feature importance analysis."""
    model = _make_xgb(xgb_params)
    model.fit(X, y)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Per-subject performance breakdown
# ──────────────────────────────────────────────────────────────────────────────

def per_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    min_n: int = 5,
) -> pd.DataFrame:
    """Decompose OOF predictions into per-subject RMSE/MAE/R²."""
    rows = []
    for sid in np.unique(groups):
        mask = groups == sid
        if mask.sum() < min_n:
            continue
        m = _metrics(y_true[mask], y_pred[mask])
        rows.append({"subject_id": int(sid), **m})
    return pd.DataFrame(rows).set_index("subject_id")
