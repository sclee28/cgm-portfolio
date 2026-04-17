"""Quick smoke-test for Day-2 modules — writes results to /tmp/day2_check.txt"""

import sys, os
sys.path.insert(0, '/home/sclee/git_repo/cgm-portfolio')

out = []

try:
    import warnings; warnings.filterwarnings('ignore')
    import pandas as pd, numpy as np

    from src.features import build_feature_matrix, FEATURE_COLS
    from src.model import (evaluate_baselines, evaluate_population_xgb,
                           evaluate_personalized_xgb, train_final_model,
                           per_subject_metrics)
    from src.config import PROCESSED_DIR

    out.append("[OK] imports")

    meal_metrics = pd.read_csv(PROCESSED_DIR / 'meal_metrics.csv')
    user_split   = pd.read_csv(PROCESSED_DIR / 'user_split.csv')
    out.append(f"[OK] CSV loaded  meal_metrics={meal_metrics.shape}  user_split={user_split.shape}")

    X, y, groups = build_feature_matrix(meal_metrics, user_split, target='iauc')
    out.append(f"[OK] feature matrix  X={X.shape}  y_mean={y.mean():.0f}  subjects={groups.nunique()}")
    out.append(f"     features: {list(X.columns)}")

    # baselines (fast)
    bl = evaluate_baselines(X, y, groups, n_splits=5)
    for r in bl:
        out.append(f"[OK] {r['model_type']:25s}  RMSE={r['rmse']:.1f}  R2={r['r2']:.3f}")

    # population xgb
    pop = evaluate_population_xgb(X, y, groups, n_splits=5)
    out.append(f"[OK] population_xgb  RMSE={pop['rmse']:.1f}  R2={pop['r2']:.3f}")

    # personalized (LOSO — slow but let it run)
    pers = evaluate_personalized_xgb(X, y, groups, verbose=False)
    out.append(f"[OK] personalized_xgb  RMSE={pers['rmse']:.1f}  R2={pers['r2']:.3f}")

    # SHAP (final model)
    import shap
    final = train_final_model(X, y)
    explainer = shap.TreeExplainer(final)
    sv = explainer.shap_values(X)
    out.append(f"[OK] SHAP values shape={sv.shape}")

    mean_abs = np.abs(sv).mean(axis=0)
    top_feat = FEATURE_COLS[np.argmax(mean_abs)]
    out.append(f"     top feature: {top_feat}  mean|SHAP|={mean_abs.max():.1f}")

    out.append("\n=== SUMMARY ===")
    out.append(f"Mean Baseline     RMSE={bl[0]['rmse']:.1f}  R2={bl[0]['r2']:.3f}")
    out.append(f"Ridge             RMSE={bl[1]['rmse']:.1f}  R2={bl[1]['r2']:.3f}")
    out.append(f"XGB Population    RMSE={pop['rmse']:.1f}  R2={pop['r2']:.3f}")
    out.append(f"XGB Personalized  RMSE={pers['rmse']:.1f}  R2={pers['r2']:.3f}")

except Exception as e:
    import traceback
    out.append(f"[ERROR] {e}")
    out.append(traceback.format_exc())

with open('/tmp/day2_check.txt', 'w') as f:
    f.write('\n'.join(out) + '\n')

print("Done")
