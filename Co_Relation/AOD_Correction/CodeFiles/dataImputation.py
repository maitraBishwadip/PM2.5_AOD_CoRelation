"""
=============================================================================
SMART IMPUTATION — AOD Proxy Study | Bangladesh Dataset
=============================================================================

WHY NOT GLOBAL MEDIAN:
  The exploratory analysis shows that missing values are NOT random (MAR,
  not MCAR). Sopura is missing 1132/5737 Wind Speed rows. Khanpur is
  missing 1164/temperature rows. These are instrument gaps at specific
  stations during specific seasons — so every missing value has a known
  spatial and seasonal context that must be used.

  Example of why global median fails:
    Agrabad  Monsoon RH mean  = 75.4%
    Baira    Winter  RH mean  = 70.7%
    Global median RH          = 71.3%
    → Imputing 71.3% for an Agrabad-Monsoon row is physically wrong by ~4%
    → Worse: Darus Salam BP varies from 992 (Monsoon) to 1014 (Winter)
      A global BP median of ~1010 would be ~18 hPa wrong for monsoon rows.

THREE IMPUTATION STRATEGIES (assigned per variable based on data structure):

  Strategy A — Stratified group median (Station × Season):
    Used for: Temperature, RH, BP
    Why: These variables have strong, stable spatial + seasonal patterns.
         The Station × Season group median is physically meaningful and
         there are enough observations in every group to be reliable.
         BP at Agrabad-Winter = 1014.4 hPa (std=2.1) — very tight,
         so the group median is an excellent imputation.

  Strategy B — KNN imputation within Station × Season group:
    Used for: Solar Rad, Wind Speed
    Why: These variables have higher within-group variance (Solar Rad std
         can be 280+ W/m² at Khulshi). KNN uses the correlated variables
         available in the same row (Temperature, RH, Rain, Month) to find
         the k=5 most similar observed rows within the same station×season
         context and takes their median. This respects both spatial context
         AND the local meteorological state of that particular day.

  Strategy C — Group median + missingness indicator flag:
    Used for: Wind_Origin (48.1% missing)
    Why: Wind_Origin is a categorical derived from trajectory analysis —
         when it is missing, it likely means the trajectory analysis could
         not confidently classify the air mass. That uncertainty is itself
         informative. So we:
           (a) impute Wind_Polluted = 0.5 (uncertain, between 0 and 1)
           (b) add a binary flag column Wind_Origin_missing = 1
         This lets the ML model learn that "uncertain wind origin" is a
         distinct regime from clearly continental or clearly marine air.

  Rain (0.9% missing, 94 rows, all in Winter):
    Imputed as 0.0 — in Winter Bangladesh, missing rain records almost
    certainly mean no rain event was recorded (dry season).

OUTPUTS:
  Imputed_Dataset_A_Raw.csv       — raw AOD + smartly imputed met
  Imputed_Dataset_B_Corrected.csv — corrected AOD features + smartly imputed met
  Imputation_QC_Report.csv        — before/after stats for every imputed column

REQUIREMENTS:
  pip install pandas numpy scikit-learn scipy

USAGE:
  python smart_imputation.py
  (run in the same folder as Master_Dataset_Final_QC.csv)
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsRegressor

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FILE  = "Master_Dataset_Final_QC.csv"
OUTPUT_A    = "Imputed_Dataset_A_Raw.csv"
OUTPUT_B    = "Imputed_Dataset_B_Corrected.csv"
OUTPUT_QC   = "Imputation_QC_Report.csv"

KNN_K       = 5     # neighbours for KNN imputation
GAMMA_RH    = 0.5   # hygroscopic growth exponent (Levy et al. 2007)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
qc_records = []

def log(msg=""):
    print(msg)

def record_qc(col, strategy, n_missing, n_total, before_mean, before_std,
              after_mean, after_std):
    qc_records.append({
        "Column"       : col,
        "Strategy"     : strategy,
        "N_missing"    : n_missing,
        "Pct_missing"  : round(n_missing / n_total * 100, 2),
        "Mean_before"  : round(before_mean, 4),
        "Std_before"   : round(before_std, 4),
        "Mean_after"   : round(after_mean, 4),
        "Std_after"    : round(after_std, 4),
        "Mean_shift"   : round(after_mean - before_mean, 4),
    })


def stratified_group_median(df, col, group_cols):
    """
    Impute missing values in `col` using the median of rows
    sharing the same values in `group_cols`.
    Falls back through progressively coarser groups if the
    finest group has no observed values.
    """
    result = df[col].copy()
    missing_mask = result.isna()

    if missing_mask.sum() == 0:
        return result

    # Build group medians at multiple granularities
    # Finest → coarsest fallback chain
    if len(group_cols) == 2:
        fallback_chains = [
            group_cols,                      # Station × Season
            [group_cols[0]],                 # Station only
            [group_cols[1]],                 # Season only
        ]
    else:
        fallback_chains = [group_cols]

    group_medians = {}
    for gc in fallback_chains:
        gm = df.groupby(gc)[col].median()
        group_medians[tuple(gc)] = gm

    # Fill missing values
    filled = 0
    for idx in df.index[missing_mask]:
        imputed = np.nan
        for gc in fallback_chains:
            key = tuple(df.loc[idx, gc].values)
            try:
                val = group_medians[tuple(gc)].loc[key]
                if not np.isnan(val):
                    imputed = val
                    break
            except KeyError:
                continue
        if np.isnan(imputed):
            # Last resort: global median
            imputed = df[col].median()
        result.loc[idx] = imputed
        filled += 1

    return result


def knn_within_group(df, target_col, feature_cols, group_cols, k=5):
    """
    For each missing value in `target_col`, find the k nearest
    observed neighbours WITHIN the same group (Station × Season),
    using `feature_cols` as predictors.
    Falls back to stratified group median if the group is too small.
    """
    result = df[target_col].copy()
    missing_mask = result.isna()

    if missing_mask.sum() == 0:
        return result

    # Normalise features to 0–1 range for KNN distance
    feat_df = df[feature_cols].copy()
    for fc in feature_cols:
        col_min = feat_df[fc].min()
        col_max = feat_df[fc].max()
        rng = col_max - col_min
        if rng > 0:
            feat_df[fc] = (feat_df[fc] - col_min) / rng
        feat_df[fc] = feat_df[fc].fillna(0.5)  # neutral for missing features

    for idx in df.index[missing_mask]:
        # Get the group of this missing row
        group_vals = {gc: df.loc[idx, gc] for gc in group_cols}
        group_filter = np.ones(len(df), dtype=bool)
        for gc, gv in group_vals.items():
            group_filter &= (df[gc] == gv).values

        observed_in_group = df.index[group_filter & result.notna()]

        if len(observed_in_group) >= k:
            # KNN within group
            X_train = feat_df.loc[observed_in_group].values
            y_train = result.loc[observed_in_group].values
            X_query = feat_df.loc[[idx]].values
            n_neighbors = min(k, len(observed_in_group))
            knn = KNeighborsRegressor(n_neighbors=n_neighbors,
                                      weights="distance")
            knn.fit(X_train, y_train)
            result.loc[idx] = knn.predict(X_query)[0]
        else:
            # Fallback: group median (Station × Season → Station → Season → global)
            imputed = np.nan
            for gc in [group_cols, [group_cols[0]], [group_cols[1]]]:
                key = tuple([group_vals[c] for c in gc])
                sub = df.loc[
                    (df[gc] == pd.Series(group_vals)).all(axis=1) & result.notna()
                ]
                if len(sub) > 0:
                    imputed = result.loc[sub.index].median()
                    break
            if np.isnan(imputed) or (isinstance(imputed, float) and
                                     np.isnan(imputed)):
                imputed = result.dropna().median()
            result.loc[idx] = imputed

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & BASIC CLEANING
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 70)
log("  SMART IMPUTATION PIPELINE")
log("  Bangladesh AOD–PM2.5 Proxy Study")
log("=" * 70)

df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
df = df.dropna(subset=["AOD", "PM2.5"])
log(f"\n[LOAD] {len(df):,} rows after dropping missing AOD/PM2.5")

# Remove BP sensor errors
df = df[(df["BP"].isna()) | (df["BP"] > 900)].copy()
log(f"[LOAD] {len(df):,} rows after removing BP < 900 hPa")

# Date features needed for grouping and KNN
df["Month"]     = df["Date"].dt.month
df["DayOfYear"] = df["Date"].dt.dayofyear
df["Year"]      = df["Date"].dt.year

# ─────────────────────────────────────────────────────────────────────────────
# 2.  RANGE FIXES BEFORE IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 1] Fixing out-of-range values before imputation")

df["Temperature"] = df["Temperature"].clip(upper=50)
df["RH"]          = df["RH"].clip(lower=0, upper=100)
df["Wind Speed"]  = df["Wind Speed"].clip(lower=0, upper=50)
rain_p99          = df["Rain"].quantile(0.99)
df["Rain"]        = df["Rain"].clip(lower=0, upper=rain_p99 * 3)
log(f"  Temperature capped at 50°C")
log(f"  RH clipped to [0, 100]%")
log(f"  Wind Speed capped at 50 m/s")
log(f"  Rain capped at {rain_p99 * 3:.1f} mm (3×p99)")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  STRATEGY A — Stratified group median (Station × Season)
#     For: Temperature, RH, BP
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 2] Strategy A: Stratified group median (Station × Season)")
log("  Variables: Temperature, RH, BP")

GROUP_COLS = ["Monitoring_Station", "Season"]

for col in ["Temperature", "RH", "BP"]:
    n_miss  = df[col].isna().sum()
    bm, bs  = df[col].mean(), df[col].std()
    df[col] = stratified_group_median(df, col, GROUP_COLS)
    am, as_ = df[col].mean(), df[col].std()
    record_qc(col, "Stratified group median (Station × Season)",
              n_miss, len(df), bm, bs, am, as_)
    log(f"  {col:15s}: {n_miss:5,} missing → filled  "
        f"(mean: {bm:.2f} → {am:.2f}, std: {bs:.2f} → {as_:.2f})")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  STRATEGY B — KNN within Station × Season group
#     For: Solar Rad, Wind Speed
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 3] Strategy B: KNN imputation within Station × Season group")
log("  Variables: Solar Rad, Wind Speed")
log(f"  k = {KNN_K} nearest neighbours (distance-weighted)")

# Features used for KNN similarity — all must be non-missing at this point
KNN_FEATURES = ["Temperature", "RH", "Rain", "Month", "DayOfYear"]

for col in ["Solar Rad", "Wind Speed"]:
    n_miss  = df[col].isna().sum()
    bm, bs  = df[col].mean(), df[col].std()
    log(f"  {col}: imputing {n_miss:,} values (this may take a moment)...")
    df[col] = knn_within_group(df, col, KNN_FEATURES, GROUP_COLS, k=KNN_K)
    am, as_ = df[col].mean(), df[col].std()
    record_qc(col, f"KNN (k={KNN_K}) within Station × Season",
              n_miss, len(df), bm, bs, am, as_)
    log(f"  {col:15s}: {n_miss:5,} missing → filled  "
        f"(mean: {bm:.2f} → {am:.2f}, std: {bs:.2f} → {as_:.2f})")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  STRATEGY C — Group median + missingness flag
#     For: Wind_Origin → Wind_Polluted (binary) + Wind_Origin_missing flag
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 4] Strategy C: Wind_Origin → binary + missing flag")

# First encode what we have
df["Wind_Polluted"] = np.where(
    df["Wind_Origin"] == "Continental (Polluted)", 1.0,
    np.where(df["Wind_Origin"] == "Marine (Clean)", 0.0, np.nan)
)
df["Wind_Origin_missing"] = df["Wind_Polluted"].isna().astype(float)

n_miss = df["Wind_Polluted"].isna().sum()
log(f"  Wind_Origin: {n_miss:,} missing ({n_miss/len(df)*100:.1f}%)")
log(f"  → Wind_Polluted: 1.0=Continental, 0.0=Marine, missing=0.5 (uncertain)")
log(f"  → Wind_Origin_missing: 1 where origin unknown (informative flag)")

df["Wind_Polluted"] = df["Wind_Polluted"].fillna(0.5)
record_qc("Wind_Polluted", "Binary encode + 0.5 for missing + flag column",
          n_miss, len(df),
          df["Wind_Polluted"].mean(), df["Wind_Polluted"].std(),
          df["Wind_Polluted"].mean(), df["Wind_Polluted"].std())

# ─────────────────────────────────────────────────────────────────────────────
# 6.  RAIN — 0.9% missing, all in Winter → fill with 0.0
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 5] Rain: 94 missing rows (all Winter) → imputed as 0.0")
n_miss = df["Rain"].isna().sum()
df["Rain"] = df["Rain"].fillna(0.0)
record_qc("Rain", "Zero-fill (all missing in dry Winter)",
          n_miss, len(df), 0.0, 0.0, df["Rain"].mean(), df["Rain"].std())

# ─────────────────────────────────────────────────────────────────────────────
# 7.  WIND DIRECTION → cyclical encoding, fill unknowns with NaN flag
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 6] Wind Dir → cyclical sin/cos + missing flag")
if "Wind Dir" in df.columns:
    df["WindDir_sin"]     = np.sin(np.deg2rad(df["Wind Dir"])).round(6)
    df["WindDir_cos"]     = np.cos(np.deg2rad(df["Wind Dir"])).round(6)
    df["WindDir_missing"] = df["Wind Dir"].isna().astype(float)
    # Fill missing with 0 (undefined direction — model will learn the flag)
    df["WindDir_sin"] = df["WindDir_sin"].fillna(0.0)
    df["WindDir_cos"] = df["WindDir_cos"].fillna(0.0)
    n_miss = df["Wind Dir"].isna().sum()
    log(f"  Wind Dir: {n_miss:,} missing → WindDir_sin/cos=0.0, WindDir_missing=1")
    df.drop(columns=["Wind Dir"], inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  ENCODE ALL REMAINING CATEGORICAL / TEXT COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 7] Encoding all categorical columns to numeric")

# Season (ordinal: pollution load order)
season_map = {"Monsoon": 1, "Post-Monsoon": 2, "Pre-Monsoon": 3, "Winter": 4}
df["Season_ord"] = df["Season"].map(season_map).fillna(2)
log(f"  Season → Season_ord  {season_map}")

# AOD_Loading (ordinal)
aod_load_map = {
    "Low AOD (<0.3)": 1, "Moderate AOD (0.3-0.8)": 2, "High AOD (>0.8)": 3
}
df["AOD_Loading_ord"] = df["AOD_Loading"].map(aod_load_map).fillna(2)
log(f"  AOD_Loading → AOD_Loading_ord")

# Rain_Status (ordinal)
rain_map = {"No Rain": 0, "Light Rain": 1, "Heavy Rain": 2}
df["Rain_Status_ord"] = df["Rain_Status"].map(rain_map).fillna(0)
log(f"  Rain_Status → Rain_Status_ord")

# Humidity_Profile (ordinal)
humidity_map = {"Dry (<50%)": 1, "Moderate (50-75%)": 2, "Humid (>75%)": 3}
df["Humidity_ord"] = df["Humidity_Profile"].map(humidity_map)
df["Humidity_ord"] = stratified_group_median(df, "Humidity_ord", GROUP_COLS)
log(f"  Humidity_Profile → Humidity_ord (missing filled by Station×Season)")

# Temp_Profile (ordinal)
temp_map = {"Cool (<20\u00b0C)": 1, "Warm (20-28\u00b0C)": 2, "Hot (>28\u00b0C)": 3}
df["Temp_ord"] = df["Temp_Profile"].map(temp_map)
df["Temp_ord"] = stratified_group_median(df, "Temp_ord", GROUP_COLS)
log(f"  Temp_Profile → Temp_ord (missing filled by Station×Season)")

# Geo_Zone and Station: target encoding (mean PM2.5)
geo_mean     = df.groupby("Geo_Zone")["PM2.5"].mean()
station_mean = df.groupby("Monitoring_Station")["PM2.5"].mean()
df["GeoZone_enc"] = df["Geo_Zone"].map(geo_mean).round(4)
df["Station_enc"] = df["Monitoring_Station"].map(station_mean).round(4)
log(f"  Geo_Zone → GeoZone_enc (mean PM2.5 per zone)")
log(f"  Monitoring_Station → Station_enc (mean PM2.5 per station)")

# Drop V Wind Speed (95.4% missing — no reliable imputation possible)
if "V Wind Speed" in df.columns:
    df.drop(columns=["V Wind Speed"], inplace=True)
    log(f"  V Wind Speed: DROPPED (95.4% missing)")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  CYCLICAL TIME FEATURES
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 8] Cyclical time encoding")

df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12).round(6)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12).round(6)
df["DOY_sin"]   = np.sin(2 * np.pi * df["DayOfYear"] / 365).round(6)
df["DOY_cos"]   = np.cos(2 * np.pi * df["DayOfYear"] / 365).round(6)
log("  Month → Month_sin, Month_cos")
log("  DayOfYear → DOY_sin, DOY_cos")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  BUILD DATASET A — Raw AOD, no physical corrections
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 9] Building Dataset A — Raw (no physical corrections)")

COLS_A = [
    "Date", "PM2.5",
    "AOD",
    "Wind Speed", "Temperature", "RH", "Solar Rad", "BP", "Rain",
    "Month", "DayOfYear", "Year",
    "Month_sin", "Month_cos", "DOY_sin", "DOY_cos",
    "WindDir_sin", "WindDir_cos", "WindDir_missing",
    "Season_ord", "AOD_Loading_ord", "Rain_Status_ord",
    "Humidity_ord", "Temp_ord",
    "Wind_Polluted", "Wind_Origin_missing",
    "GeoZone_enc", "Station_enc",
    "Latitude", "Longitude",
]
COLS_A = [c for c in COLS_A if c in df.columns]
df_A   = df[COLS_A].copy()

log(f"  Columns: {len(COLS_A)}")
log(f"  Shape  : {df_A.shape}")
log(f"  Any NaN: {df_A.drop(columns=['Date']).isnull().any().any()}")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  PHYSICAL CORRECTIONS → Dataset B
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 10] Computing physical corrections for Dataset B")

# AOD nonlinear transforms
df["AOD_log"]  = np.log1p(df["AOD"].clip(lower=0)).round(6)
df["AOD_sqrt"] = np.sqrt(df["AOD"].clip(lower=0)).round(6)
df["AOD_sq"]   = (df["AOD"] ** 2).round(6)
log("  AOD_log, AOD_sqrt, AOD_sq")

# Boundary Layer Height proxy
# Temperature / (Wind_Speed + 0.1) — now using properly imputed Wind Speed
df["BL_proxy"] = (df["Temperature"] / (df["Wind Speed"] + 0.1)).round(6)
log(f"  BL_proxy = Temp / (WS + 0.1)  "
    f"[median={df['BL_proxy'].median():.2f}, max={df['BL_proxy'].max():.2f}]")

# Hygroscopic growth factor — using properly imputed RH
df["f_RH"] = ((1 - df["RH"] / 100).clip(lower=0.01) ** GAMMA_RH).round(6)
log(f"  f_RH = (1 - RH/100)^{GAMMA_RH}  "
    f"[median={df['f_RH'].median():.4f}]")

# Corrected AOD features
df["AOD_BLH_corr"]  = (df["AOD"] / df["BL_proxy"]).round(6)
df["AOD_RH_corr"]   = (df["AOD"] * df["f_RH"]).round(6)
df["AOD_FULL_corr"] = (df["AOD"] * df["f_RH"] / df["BL_proxy"]).round(6)
log("  AOD_BLH_corr  = AOD / BL_proxy")
log("  AOD_RH_corr   = AOD × f_RH")
log("  AOD_FULL_corr = AOD × f_RH / BL_proxy  ← full correction")

# Physical interaction terms — now with properly imputed met variables
df["Wet_scavenge"]   = (df["Rain"]      * df["AOD"]).round(6)
df["AOD_x_RH"]       = (df["RH"]        * df["AOD"]).round(6)
df["AOD_x_Temp"]     = (df["Temperature"] * df["AOD"]).round(6)
df["AOD_x_WS"]       = (df["Wind Speed"] * df["AOD"]).round(6)
df["AOD_x_SolRad"]   = (df["Solar Rad"]  * df["AOD"]).round(6)
df["AOD_x_Season"]   = (df["Season_ord"] * df["AOD"]).round(6)
df["AOD_x_WindPol"]  = (df["Wind_Polluted"] * df["AOD"]).round(6)
log("  Physical interactions: AOD×RH, AOD×Temp, AOD×WS, AOD×SolRad, "
    "AOD×Season, AOD×WindPolluted, Wet_scavenge")

# ─────────────────────────────────────────────────────────────────────────────
# 12.  BUILD DATASET B — Corrected
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 11] Building Dataset B — Corrected")

COLS_B = [
    "Date", "PM2.5",
    "AOD", "AOD_log", "AOD_sqrt", "AOD_sq",
    "AOD_BLH_corr", "AOD_RH_corr", "AOD_FULL_corr",
    "BL_proxy", "f_RH",
    "Wind Speed", "Temperature", "RH", "Solar Rad", "BP", "Rain",
    "Wet_scavenge",
    "AOD_x_RH", "AOD_x_Temp", "AOD_x_WS", "AOD_x_SolRad",
    "AOD_x_Season", "AOD_x_WindPol",
    "Month", "DayOfYear", "Year",
    "Month_sin", "Month_cos", "DOY_sin", "DOY_cos",
    "WindDir_sin", "WindDir_cos", "WindDir_missing",
    "Season_ord", "AOD_Loading_ord", "Rain_Status_ord",
    "Humidity_ord", "Temp_ord",
    "Wind_Polluted", "Wind_Origin_missing",
    "GeoZone_enc", "Station_enc",
    "Latitude", "Longitude",
]
COLS_B = [c for c in COLS_B if c in df.columns]
df_B   = df[COLS_B].copy()

log(f"  Columns: {len(COLS_B)}")
log(f"  Shape  : {df_B.shape}")
log(f"  Any NaN: {df_B.drop(columns=['Date']).isnull().any().any()}")

# ─────────────────────────────────────────────────────────────────────────────
# 13.  SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 12] Saving outputs")

df_A.to_csv(OUTPUT_A, index=False)
df_B.to_csv(OUTPUT_B, index=False)

qc_df = pd.DataFrame(qc_records)
qc_df.to_csv(OUTPUT_QC, index=False)

log(f"  {OUTPUT_A}  →  {df_A.shape[0]:,} × {df_A.shape[1]}")
log(f"  {OUTPUT_B}  →  {df_B.shape[0]:,} × {df_B.shape[1]}")
log(f"  {OUTPUT_QC} →  imputation QC stats per column")

# ─────────────────────────────────────────────────────────────────────────────
# 14.  SANITY REPORT
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("  IMPUTATION SANITY REPORT")
log("=" * 70)
print(qc_df[["Column","Strategy","N_missing","Pct_missing",
             "Mean_before","Mean_after","Mean_shift"]].to_string(index=False))

from scipy import stats
log("\nSpearman rho with PM2.5 — before vs after correction:")
check = [
    ("AOD (raw)",     "AOD"),
    ("AOD_FULL_corr", "AOD_FULL_corr"),
    ("AOD_BLH_corr",  "AOD_BLH_corr"),
    ("AOD_RH_corr",   "AOD_RH_corr"),
    ("BL_proxy",      "BL_proxy"),
    ("f_RH",          "f_RH"),
    ("Temperature",   "Temperature"),
    ("RH",            "RH"),
]
for label, col in check:
    if col in df.columns:
        rho, p = stats.spearmanr(df[col], df["PM2.5"])
        log(f"  {label:22s}: rho={rho:+.4f}  p={p:.3e}")

log("\n" + "=" * 70)
log("  Done. Feed these two files to your ML training script:")
log(f"  Baseline model  → {OUTPUT_A}")
log(f"  Corrected model → {OUTPUT_B}")
log("=" * 70)