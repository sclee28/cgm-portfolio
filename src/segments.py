"""K-Means segmentation pipeline for metabolic user profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


SEGMENT_FEATURES = ["tir", "cv_pct", "mean_glucose", "gmi", "hbgi", "lbgi"]

CLUSTER_LABELS: dict[int, dict] = {
    # populated at runtime based on centroid ordering
}


def build_segment_matrix(
    subject_summary: pd.DataFrame,
    user_split: pd.DataFrame,
    features: list[str] = SEGMENT_FEATURES,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge subject CGM summary with demographics and return segmentation feature matrix.

    Returns
    -------
    X_raw : un-scaled DataFrame (for display)
    meta  : full merged DataFrame including demographics
    """
    meta = subject_summary.reset_index().merge(
        user_split[["subject_id", "age", "bmi", "hba1c", "gender"]],
        on="subject_id",
        how="left",
    )
    X_raw = meta[features].copy()
    if drop_na:
        valid_mask = X_raw.notna().all(axis=1)
        meta = meta[valid_mask].reset_index(drop=True)
        X_raw = X_raw[valid_mask].reset_index(drop=True)
    return X_raw, meta


def fit_kmeans(
    X_raw: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 20,
) -> tuple[np.ndarray, StandardScaler, KMeans]:
    """Scale features and fit K-Means.

    Returns
    -------
    X_scaled : scaled array
    scaler   : fitted StandardScaler
    km       : fitted KMeans model
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.values)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    km.fit(X_scaled)
    return X_scaled, scaler, km


def order_clusters_by_tir(
    km: KMeans,
    scaler: StandardScaler,
    features: list[str] = SEGMENT_FEATURES,
) -> dict[int, int]:
    """Return a mapping {raw_label → ordered_label} sorted by TIR descending.

    Cluster with highest TIR = cluster 0 ("Stable"), lowest = last ("At-risk").
    """
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    tir_idx = features.index("tir")
    tir_vals = centroids_orig[:, tir_idx]
    rank = np.argsort(-tir_vals)  # descending TIR
    return {int(old): int(new) for new, old in enumerate(rank)}


SEGMENT_NAMES = {
    0: "Stable",
    1: "Moderate",
    2: "At-risk",
}

SEGMENT_COLORS = {
    0: "#4CAF50",   # green
    1: "#FF9800",   # orange
    2: "#F44336",   # red
}

SEGMENT_COACHING = {
    0: "혈당이 전반적으로 안정적입니다. 현재 식습관을 유지하세요.",
    1: "혈당 변동성이 보통 수준입니다. 탄수화물 섭취량과 식사 타이밍을 점검해보세요.",
    2: "혈당 불안정 패턴이 관찰됩니다. 식단 조정과 전문가 상담을 권장합니다.",
}
