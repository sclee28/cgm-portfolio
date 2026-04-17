"""Project-wide constants and column name mappings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data" / "CGMacros"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ──────────────────────────────────────────────
# bio.csv  →  clean column names
# ──────────────────────────────────────────────
COL_MAP_BIO: dict[str, str] = {
    "subject": "subject_id",
    "Age": "age",
    "Gender": "gender",
    "BMI": "bmi",
    "Body weight ": "weight_lbs",
    "Height ": "height_in",
    "Self-identify ": "ethnicity",
    "A1c PDL (Lab)": "hba1c",
    "Fasting GLU - PDL (Lab)": "fasting_glucose",
    "Insulin ": "insulin",
    "Triglycerides": "triglycerides",
    "Cholesterol": "cholesterol",
    "HDL": "hdl",
    "Non HDL ": "non_hdl",
    "LDL (Cal)": "ldl",
    "VLDL (Cal)": "vldl",
    "Cho/HDL Ratio": "cho_hdl_ratio",
}

# ──────────────────────────────────────────────
# CGMacros-XXX.csv  →  clean column names
# ──────────────────────────────────────────────
COL_MAP_CGM: dict[str, str] = {
    "Timestamp": "timestamp",
    "Libre GL": "libre_gl",
    "Dexcom GL": "dexcom_gl",
    "HR": "hr",
    "Calories (Activity)": "activity_calories",
    "METs": "mets",
    "Meal Type": "meal_type",
    "Calories": "meal_calories",
    "Carbs": "carbs",
    "Protein": "protein",
    "Fat": "fat",
    "Fiber": "fiber",
    "Amount Consumed ": "amount_consumed",
    "Image path": "image_path",
}

# ──────────────────────────────────────────────
# Blood glucose thresholds  (mg/dL)
# ──────────────────────────────────────────────
GLUCOSE_LOW: float = 70.0       # TBR cutoff (Level 1)
GLUCOSE_HIGH: float = 140.0     # TAR cutoff (healthy adult)
GLUCOSE_VERY_LOW: float = 54.0  # Level 2 hypoglycemia

# Post-prandial analysis window
POSTPRANDIAL_WINDOW: int = 120   # minutes

# Recovery time: within ±threshold of premeal baseline = "recovered"
RECOVERY_THRESHOLD: float = 5.0  # mg/dL
