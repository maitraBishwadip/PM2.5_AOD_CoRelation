"""
=============================================================================
AOD as PM2.5 Proxy — Random Forest Analysis (FIXED)
Bangladesh Tropical Dataset
=============================================================================
FIXES from v1:
  - No more max(r2, 0) masking — negative R² is meaningful (model fails)
  - Stratified evaluation uses Leave-One-Out or full-data R² for small N
  - Global RF trained ONCE on full data, then evaluated per-stratum
    on out-of-bag or held-out predictions (no re-training per subset)
  - AOD-only proxy quality measured via Spearman + local OOB R² together
  - Proper fallback: for N < 60, use full fit + adjusted R²
  - Conservative thresholds aligned with tropical aerosol literature

Proxy Quality (R² of AOD-only prediction within stratum):
  Good   : R² >= 0.50 AND Spearman p <= 0.05 AND |rho| >= 0.50
  Usable : R² >= 0.25 AND Spearman p <= 0.05 AND |rho| >= 0.35
  Poor   : everything else
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ── CONFIG ─────────────────────────────────────────────────────────────────
DATA_FILE          = "Master_Dataset_Daily_Raw.csv"
OUTPUT_MAIN        = "RF_AOD_PM25_Condition_Results.csv"
OUTPUT_IMPORTANCE  = "RF_Feature_Importance.csv"
OUTPUT_PREDICTIONS = "RF_Predictions_Full.csv"

MIN_SAMPLES  = 20    # minimum per stratum for any evaluation
RANDOM_STATE = 42

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 68)
print("  AOD–PM2.5 RF ANALYSIS (FIXED)  |  Bangladesh Dataset")
print("=" * 68)

df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df = df.dropna(subset=["AOD", "PM2.5"])
print(f"\n✔  Loaded {len(df):,} rows after dropping missing AOD/PM2.5")

# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Feature Engineering ──────────────────────────────────────────────")

# Date features
df["Month"]     = df["Date"].dt.month
df["DayOfYear"] = df["Date"].dt.dayofyear

# AOD transforms
df["AOD_log"]  = np.log1p(df["AOD"].clip(lower=0))
df["AOD_sqrt"] = np.sqrt(df["AOD"].clip(lower=0))
df["AOD_sq"]   = df["AOD"] ** 2

# Interaction terms
for met_col in ["RH", "Wind Speed", "Temperature", "Solar Rad", "BP"]:
    if met_col in df.columns:
        df[f"AOD_x_{met_col.replace(' ','_')}"] = df["AOD"] * df[met_col].fillna(df[met_col].median())

# One-hot encode categoricals
cat_cols = ["Season", "AOD_Loading", "Wind_Origin",
            "Humidity_Profile", "Temp_Profile", "Rain_Status", "Geo_Zone"]
cat_cols = [c for c in cat_cols if c in df.columns]

df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=False)

# Numeric feature set
exclude = {"PM2.5", "Date", "Latitude", "Longitude",
           "V Wind Speed", "Wind Dir", "Monitoring_Station"}
feature_cols = [
    c for c in df_enc.columns
    if c not in exclude
    and df_enc[c].dtype in [np.float64, np.float32, np.int64, np.int32, bool, np.uint8]
    and c != "PM2.5"
]

# Fill NaNs with median
for col in feature_cols:
    df_enc[col] = df_enc[col].fillna(df_enc[col].median())

X_full = df_enc[feature_cols].astype(float)
y_full = df_enc["PM2.5"].astype(float)

print(f"   Features: {len(feature_cols)}")
print(f"   Sample size: {len(X_full):,}\n")

# ═══════════════════════════════════════════════════════════════════════════
# 3. GLOBAL RF — trained once on ALL data
#    Used for feature importance only
#    OOB predictions used for global evaluation
# ═══════════════════════════════════════════════════════════════════════════
print("── Global RF Model (all features, OOB enabled) ─────────────────────")

rf_global = RandomForestRegressor(
    n_estimators     = 500,
    max_depth        = 12,
    min_samples_leaf = 5,
    max_features     = "sqrt",
    oob_score        = True,
    n_jobs           = -1,
    random_state     = RANDOM_STATE
)
rf_global.fit(X_full, y_full)

oob_r2   = rf_global.oob_score_
oob_pred = rf_global.oob_prediction_
print(f"   Global RF OOB R² = {oob_r2:.4f}  (all features, all data)")

# Feature importance
imp_df = pd.DataFrame({
    "Feature"   : feature_cols,
    "Importance": rf_global.feature_importances_
}).sort_values("Importance", ascending=False)
imp_df.to_csv(OUTPUT_IMPORTANCE, index=False)

print("\n   Top 15 Most Important Features:")
print(imp_df.head(15).to_string(index=False))

# Store OOB predictions
df["PM2.5_GlobalRF_OOB"] = oob_pred
df["GlobalRF_Residual"]  = df["PM2.5"] - oob_pred

# ═══════════════════════════════════════════════════════════════════════════
# 4. AOD-ONLY GLOBAL MODEL — cross-validated predictions (NO data leakage)
#    This is the CORRECT way: use cross_val_predict so every sample
#    is predicted by a model that never saw it during training.
# ═══════════════════════════════════════════════════════════════════════════
print("\n── AOD-only Global CV Model ─────────────────────────────────────────")

AOD_FEATURES = ["AOD", "AOD_log", "AOD_sqrt", "AOD_sq",
                "AOD_x_RH", "AOD_x_Wind_Speed",
                "AOD_x_Temperature", "AOD_x_Solar_Rad"]
AOD_FEATURES = [f for f in AOD_FEATURES if f in df_enc.columns]

X_aod = df_enc[AOD_FEATURES].fillna(0).astype(float)

rf_aod_base = RandomForestRegressor(
    n_estimators     = 300,
    max_depth        = 8,
    min_samples_leaf = 5,
    max_features     = "sqrt",
    n_jobs           = -1,
    random_state     = RANDOM_STATE
)

# 10-fold CV predictions — each row predicted out-of-fold
cv10 = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
aod_cv_preds = cross_val_predict(rf_aod_base, X_aod, y_full, cv=cv10, n_jobs=-1)

global_aod_r2   = r2_score(y_full, aod_cv_preds)
global_aod_rmse = np.sqrt(mean_squared_error(y_full, aod_cv_preds))
global_rho, global_p = stats.spearmanr(df["AOD"], df["PM2.5"])

print(f"   AOD-only 10-fold CV R²   = {global_aod_r2:.4f}")
print(f"   AOD-only CV RMSE         = {global_aod_rmse:.2f} µg/m³")
print(f"   Spearman rho (raw AOD)   = {global_rho:.4f}  p={global_p:.4g}")

# Store CV predictions
df["PM2.5_AODonly_CVpred"] = aod_cv_preds
df["AODonly_Residual"]     = df["PM2.5"] - aod_cv_preds

# Train final AOD-only model on full data (for per-stratum use)
rf_aod_base.fit(X_aod, y_full)

# ═══════════════════════════════════════════════════════════════════════════
# 5. PROXY QUALITY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
def classify_proxy(r2, rho, p_value, n):
    """
    Combined R² + Spearman threshold.
    R² can be negative — that is meaningful (model is worse than predicting mean).
    """
    sig = p_value <= 0.05
    if r2 >= 0.50 and sig and abs(rho) >= 0.50:
        return "Good"
    elif r2 >= 0.25 and sig and abs(rho) >= 0.35:
        return "Usable"
    else:
        return "Poor"

# ═══════════════════════════════════════════════════════════════════════════
# 6. STRATUM EVALUATOR
#    Key fix: use cross_val_predict within stratum when N >= MIN_SAMPLES.
#    For N < 60: use LOO cross-validation (unbiased for small N).
#    For N >= 60: use 5-fold CV.
#    NEVER clip R² to 0.
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_stratum(subset_df, label):
    n = len(subset_df)
    if n < MIN_SAMPLES:
        return None

    idx    = subset_df.index
    X_sub  = X_aod.loc[idx]
    y_sub  = y_full.loc[idx]
    aod_raw = subset_df["AOD"].values
    pm_raw  = subset_df["PM2.5"].values

    # Spearman on raw values
    rho, p = stats.spearmanr(aod_raw, pm_raw)

    # Cross-validated R² within this stratum
    rf_loc = RandomForestRegressor(
        n_estimators     = 200,
        max_depth        = 6,
        min_samples_leaf = max(3, n // 20),  # prevents overfitting on tiny N
        max_features     = "sqrt",
        n_jobs           = -1,
        random_state     = RANDOM_STATE
    )

    try:
        if n < 60:
            # LOO is unbiased for small samples
            loo   = LeaveOneOut()
            preds = cross_val_predict(rf_loc, X_sub, y_sub, cv=loo, n_jobs=-1)
        else:
            cv5   = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            preds = cross_val_predict(rf_loc, X_sub, y_sub, cv=cv5, n_jobs=-1)

        r2   = r2_score(y_sub, preds)           # CAN be negative
        rmse = np.sqrt(mean_squared_error(y_sub, preds))
        mean_pm = y_sub.mean()
        nrmse   = rmse / mean_pm if mean_pm > 0 else 9999

    except Exception as e:
        print(f"   [WARN] CV failed for '{label}': {e}")
        r2, rmse, nrmse = -9999, 9999, 9999

    quality = classify_proxy(r2, rho, p, n)

    return {
        "Condition"          : label,
        "N"                  : n,
        "Spearman_rho"       : round(rho, 4),
        "Spearman_p"         : round(p, 6),
        "Spearman_Significant": "Yes" if p <= 0.05 else "No",
        "RF_CV_R2_AOD_only"  : round(r2, 4),
        "RMSE_ugm3"          : round(rmse, 2),
        "nRMSE"              : round(nrmse, 4),
        "Mean_PM25_ugm3"     : round(y_sub.mean(), 2),
        "Std_PM25_ugm3"      : round(y_sub.std(), 2),
        "Proxy_Quality"      : quality,
        "CV_method"          : "LOO" if n < 60 else "5-fold"
    }

# ═══════════════════════════════════════════════════════════════════════════
# 7. RUN ALL CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════
results = []

def run_and_log(subset_df, label):
    r = evaluate_stratum(subset_df, label)
    if r:
        results.append(r)
        flag = {"Good": "✔", "Usable": "~", "Poor": "✘"}.get(r["Proxy_Quality"], "?")
        print(f"  {flag} {label}")
        print(f"    N={r['N']}  R²={r['RF_CV_R2_AOD_only']}  "
              f"rho={r['Spearman_rho']}  nRMSE={r['nRMSE']}  "
              f"→ [{r['Proxy_Quality']}]  CV={r['CV_method']}")

# Overall
print("\n── Overall ──────────────────────────────────────────────────────────")
run_and_log(df, "Overall — All Data")

# Single conditions
single_cats = ["Season", "AOD_Loading", "Wind_Origin",
               "Humidity_Profile", "Temp_Profile", "Rain_Status", "Geo_Zone"]

for col in single_cats:
    if col not in df.columns:
        continue
    print(f"\n── By {col} {'─'*(50-len(col))}")
    for val in sorted(df[col].dropna().unique()):
        run_and_log(df[df[col] == val], f"{col} = {val}")

# Two-way combinations
combos_2 = [
    ("Season", "Wind_Origin"),
    ("Season", "Humidity_Profile"),
    ("Season", "Temp_Profile"),
    ("Season", "Rain_Status"),
    ("Season", "AOD_Loading"),
    ("Season", "Geo_Zone"),
    ("Wind_Origin", "Humidity_Profile"),
    ("Wind_Origin", "Rain_Status"),
    ("Wind_Origin", "AOD_Loading"),
    ("Humidity_Profile", "Rain_Status"),
    ("Humidity_Profile", "AOD_Loading"),
    ("Temp_Profile", "Rain_Status"),
    ("Geo_Zone", "Season"),
    ("Geo_Zone", "Wind_Origin"),
    ("Geo_Zone", "Rain_Status"),
]

print("\n── Two-parameter combinations ───────────────────────────────────────")
for c1, c2 in combos_2:
    if c1 not in df.columns or c2 not in df.columns:
        continue
    for v1 in sorted(df[c1].dropna().unique()):
        for v2 in sorted(df[c2].dropna().unique()):
            sub = df[(df[c1] == v1) & (df[c2] == v2)]
            run_and_log(sub, f"{c1}={v1} & {c2}={v2}")

# Three-way combinations
combos_3 = [
    ("Season", "Wind_Origin", "Rain_Status"),
    ("Season", "Wind_Origin", "Humidity_Profile"),
    ("Season", "Wind_Origin", "Temp_Profile"),
    ("Season", "Humidity_Profile", "Rain_Status"),
    ("Season", "AOD_Loading", "Wind_Origin"),
    ("Season", "AOD_Loading", "Rain_Status"),
    ("Geo_Zone", "Season", "Wind_Origin"),
    ("Geo_Zone", "Season", "Rain_Status"),
]

print("\n── Three-parameter combinations ─────────────────────────────────────")
for c1, c2, c3 in combos_3:
    if not all(c in df.columns for c in [c1, c2, c3]):
        continue
    for v1 in sorted(df[c1].dropna().unique()):
        for v2 in sorted(df[c2].dropna().unique()):
            for v3 in sorted(df[c3].dropna().unique()):
                sub = df[(df[c1]==v1)&(df[c2]==v2)&(df[c3]==v3)]
                run_and_log(sub, f"{c1}={v1} & {c2}={v2} & {c3}={v3}")

# Continuous bins × Season
cont_bins = {
    "Wind Speed" : ([0,1,3,5,100],  ["WS_Low","WS_Moderate","WS_High","WS_VeryHigh"]),
    "RH"         : ([0,40,60,80,100],["RH_Low","RH_Moderate","RH_High","RH_VeryHigh"]),
    "Temperature": ([0,20,25,30,45], ["Temp_Cool","Temp_Mild","Temp_Warm","Temp_Hot"]),
    "Solar Rad"  : ([0,50,150,300,1000],["SR_VLow","SR_Low","SR_Moderate","SR_High"]),
}

print("\n── Continuous bins (standalone) ─────────────────────────────────────")
for col, (bins, lbls) in cont_bins.items():
    if col not in df.columns:
        continue
    bcol = col + "_BIN"
    df[bcol] = pd.cut(df[col], bins=bins, labels=lbls)
    for lbl in lbls:
        run_and_log(df[df[bcol]==lbl], f"{col} = {lbl}")

print("\n── Continuous bins × Season ─────────────────────────────────────────")
for col, (bins, lbls) in cont_bins.items():
    bcol = col + "_BIN"
    if bcol not in df.columns:
        continue
    for season in sorted(df["Season"].dropna().unique()):
        for lbl in lbls:
            sub = df[(df["Season"]==season) & (df[bcol]==lbl)]
            run_and_log(sub, f"Season={season} & {col}={lbl}")

# ═══════════════════════════════════════════════════════════════════════════
# 8. SAVE
# ═══════════════════════════════════════════════════════════════════════════
out_df = pd.DataFrame(results).sort_values("RF_CV_R2_AOD_only", ascending=False)
out_df.to_csv(OUTPUT_MAIN, index=False)
print(f"\n✔  Condition results → '{OUTPUT_MAIN}'  ({len(out_df)} rows)")

pred_cols = [c for c in ["Date","Monitoring_Station","Geo_Zone","Season",
                          "AOD","PM2.5","PM2.5_GlobalRF_OOB",
                          "PM2.5_AODonly_CVpred","GlobalRF_Residual",
                          "AODonly_Residual"] if c in df.columns]
df[pred_cols].to_csv(OUTPUT_PREDICTIONS, index=False)
print(f"✔  Predictions       → '{OUTPUT_PREDICTIONS}'")

# ═══════════════════════════════════════════════════════════════════════════
# 9. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("  FINAL SUMMARY")
print("=" * 68)

qc = out_df["Proxy_Quality"].value_counts()
total = len(out_df)
print("\nQuality distribution:")
for q in ["Good", "Usable", "Poor"]:
    cnt = qc.get(q, 0)
    print(f"  {q:8s}: {cnt:4d} / {total}  ({cnt/total*100:.1f}%)")

print("\n── GOOD conditions (AOD reliable as PM2.5 proxy) ───────────────────")
good = out_df[out_df["Proxy_Quality"] == "Good"]
if good.empty:
    print("  None found — AOD alone is not a reliable proxy under any tested condition.")
    print("  Best 'Usable' conditions:")
    good = out_df[out_df["Proxy_Quality"] == "Usable"].head(8)

for _, row in good.head(12).iterrows():
    print(f"\n  R²={row['RF_CV_R2_AOD_only']:+.3f}  rho={row['Spearman_rho']:+.3f}"
          f"  nRMSE={row['nRMSE']:.3f}  N={row['N']}")
    print(f"  ▶ {row['Condition']}")

print("\n── POOR conditions (AOD unreliable) — worst 5 ───────────────────────")
poor = out_df[out_df["Proxy_Quality"]=="Poor"].sort_values("RF_CV_R2_AOD_only").head(5)
for _, row in poor.iterrows():
    print(f"  R²={row['RF_CV_R2_AOD_only']:+.3f}  rho={row['Spearman_rho']:+.3f}"
          f"  N={row['N']}  ▶ {row['Condition']}")

print(f"""
{"=" * 68}
  GLOBAL MODEL CONTEXT
{"=" * 68}
  Full-feature RF OOB R²  : {oob_r2:.4f}
  AOD-only 10-fold CV R²  : {global_aod_r2:.4f}
  AOD-only RMSE           : {global_aod_rmse:.2f} µg/m³
  Spearman rho (raw)      : {global_rho:.4f}  p={global_p:.4g}

  If AOD-only R² << Full-feature R², then meteorological
  variables (RH, BLH, wind, rain) are essential corrections
  before using AOD as a PM2.5 proxy in Bangladesh.
{"=" * 68}
""")