"""Feature engineering for post-prandial glucose response (PPGR) prediction."""

import pandas as pd

MEAL_TYPE_MAP: dict[str, int] = {
    "Breakfast": 0,
    "Lunch": 1,
    "Dinner": 2,
    "Snacks": 3,
}

FEATURE_COLS: list[str] = [
    "carbs",
    "protein",
    "fat",
    "fiber",
    "meal_calories",
    "meal_type_enc",
    "hour",
    "age",
    "bmi",
    "hba1c",
    "gender_enc",
]

TARGET_IAUC = "iauc"
TARGET_PPGE = "ppge"


def build_feature_matrix(
    meal_metrics: pd.DataFrame,
    user_split: pd.DataFrame,
    target: str = TARGET_IAUC,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build feature matrix X, target y, and subject groups from precomputed metrics.

    Parameters
    ----------
    meal_metrics : output of compute_meal_metrics() — one row per meal event.
    user_split   : output of EDA step — one row per subject with demographics.
    target       : 'iauc' (default) or 'ppge'.
    drop_na      : drop rows with any NaN in features or target.

    Returns
    -------
    X      : pd.DataFrame of shape (n_meals, n_features)
    y      : pd.Series of target values
    groups : pd.Series of subject_id (for GroupKFold)
    """
    df = meal_metrics.copy()

    subj_cols = ["subject_id", "age", "bmi", "hba1c", "gender"]
    df = df.merge(user_split[subj_cols], on="subject_id", how="left")

    df["meal_time"] = pd.to_datetime(df["meal_time"])
    df["hour"] = df["meal_time"].dt.hour

    df["meal_type_enc"] = df["meal_type"].map(MEAL_TYPE_MAP).fillna(3).astype(int)
    df["gender_enc"] = (df["gender"] == "M").astype(int)

    df[target] = df[target].clip(lower=0)

    keep = ["subject_id"] + FEATURE_COLS + [target]
    feature_df = df[keep].copy()
    if drop_na:
        feature_df = feature_df.dropna()

    feature_df = feature_df.reset_index(drop=True)
    X = feature_df[FEATURE_COLS]
    y = feature_df[target]
    groups = feature_df["subject_id"]

    return X, y, groups


FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "carbs": "Carbohydrates (g)",
    "protein": "Protein (g)",
    "fat": "Fat (g)",
    "fiber": "Fiber (g)",
    "meal_calories": "Meal Calories (kcal)",
    "meal_type_enc": "Meal Type",
    "hour": "Hour of Day",
    "age": "Age (years)",
    "bmi": "BMI",
    "hba1c": "HbA1c (%)",
    "gender_enc": "Sex (male=1)",
}
