"""
=============================================================================
DATASET CREATION + CORRELATION ANALYSIS
AOD Proxy Study | Bangladesh Dataset
=============================================================================

WHAT THIS SCRIPT DOES:
  Produces TWO clean datasets and a full correlation analysis:

  DATASET A — Cleaned only (no feature engineering)
    - All text columns removed or encoded to numeric
    - Missing values imputed using context-aware strategies
    - No new derived features, no transforms
    - This is the baseline: raw cleaned data as-is

  DATASET B — Cleaned + Physical Feature Engineering
    - Everything in A PLUS:
      - BL_proxy, f_RH, AOD_FULL_corr (Method 1 corrections)
      - AOD nonlinear transforms
      - Physical interaction terms
      - Cyclical time features

  CORRELATION ANALYSIS (run on Dataset A)
    - Spearman correlation matrix (full)
    - Pearson correlation matrix (full)
    - Individual correlations of every variable with PM2.5 and AOD
    - All results saved as CSV
    - All plots saved as PNG

IMPUTATION STRATEGIES (context-aware, not global median):
  Temperature, RH, BP     → Stratified group median (Station × Season)
                             Falls back: Station → Season → global median
  Solar Rad, Wind Speed   → KNN (k=5) within Station × Season group
                             using available correlated variables as features
                             Falls back to group median if group too small
  Rain (94 missing, Winter) → 0.0 (dry season, confirmed by season pattern)
  Wind_Origin             → Binary (Continental=1, Marine=0) + 0.5 for missing
                             + Wind_Origin_missing flag column
  Humidity_Profile,
  Temp_Profile,
  Rain_Status             → Ordinal integers, missing filled by group median
  V Wind Speed            → DROPPED (95.4% missing)
  Wind Dir                → sin/cos cyclical, 0.0 where missing + flag

OUTPUTS (all saved next to this script):
  Cleaned_Dataset_A.csv           — cleaned only, no feature engineering
  Cleaned_Dataset_B.csv           — cleaned + physical corrections
  Correlation_Results.csv         — all pairwise correlations with PM2.5 & AOD
  Correlation_Matrix_Spearman.csv — full Spearman matrix
  Correlation_Matrix_Pearson.csv  — full Pearson matrix
  Correlation_Plots/              — all PNG figures

REQUIREMENTS:
  pip install pandas numpy matplotlib seaborn scipy scikit-learn

USAGE:
  Place in same folder as Master_Dataset_Final_QC.csv
  python dataset_creation_correlation.py
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — all paths relative to this script
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "Master_Dataset_Final_QC.csv")
PLOT_DIR   = os.path.join(SCRIPT_DIR, "Correlation_Plots")
DPI        = 150
KNN_K      = 5
GAMMA_RH   = 0.5   # hygroscopic growth exponent (Levy et al. 2007)

os.makedirs(PLOT_DIR, exist_ok=True)

SEASON_ORDER  = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
SEASON_COLORS = {
    "Winter"      : "#2166ac",
    "Pre-Monsoon" : "#f4a582",
    "Monsoon"     : "#1a9641",
    "Post-Monsoon": "#d6604d",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_fig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [PLOT] Saved → {path}")


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def stratified_group_median(series, df, group_cols):
    """
    Impute missing values using the median of the Station × Season group.
    Fallback chain: Station×Season → Station → Season → global median.
    """
    result = series.copy()
    missing_idx = result.index[result.isna()]
    if len(missing_idx) == 0:
        return result

    col = series.name

    # Pre-compute medians at each level
    g2 = df.groupby(group_cols)[col].median()          # Station × Season
    g1a = df.groupby(group_cols[0])[col].median()      # Station only
    g1b = df.groupby(group_cols[1])[col].median()      # Season only
    g0  = df[col].median()                              # global

    for idx in missing_idx:
        key2 = (df.loc[idx, group_cols[0]], df.loc[idx, group_cols[1]])
        key1a = df.loc[idx, group_cols[0]]
        key1b = df.loc[idx, group_cols[1]]

        val = g2.get(key2, np.nan)
        if np.isnan(val):
            val = g1a.get(key1a, np.nan)
        if np.isnan(val):
            val = g1b.get(key1b, np.nan)
        if np.isnan(val):
            val = g0

        result.loc[idx] = val

    return result


def knn_within_group(df, target_col, feature_cols, group_cols, k=5):
    """
    For each missing value in target_col, find k nearest observed neighbours
    WITHIN the same Station × Season group using feature_cols as predictors.
    Fallback: stratified group median if group is too small for KNN.
    """
    result = df[target_col].copy()
    missing_idx = result.index[result.isna()]
    if len(missing_idx) == 0:
        return result

    # Normalise features to [0,1] for fair KNN distances
    feat_df = df[feature_cols].copy()
    for fc in feature_cols:
        col_min = feat_df[fc].min()
        col_max = feat_df[fc].max()
        rng = col_max - col_min
        if rng > 0:
            feat_df[fc] = (feat_df[fc] - col_min) / rng
        feat_df[fc] = feat_df[fc].fillna(0.5)

    for idx in missing_idx:
        g_vals = {gc: df.loc[idx, gc] for gc in group_cols}

        # Build group mask
        mask = pd.Series(True, index=df.index)
        for gc, gv in g_vals.items():
            mask &= (df[gc] == gv)

        observed = result.index[mask & result.notna()]

        if len(observed) >= k:
            X_train = feat_df.loc[observed].values
            y_train = result.loc[observed].values
            X_query = feat_df.loc[[idx]].values
            n_nb    = min(k, len(observed))
            knn     = KNeighborsRegressor(n_neighbors=n_nb, weights="distance")
            knn.fit(X_train, y_train)
            result.loc[idx] = knn.predict(X_query)[0]
        else:
            # Fallback: stratified group median
            result = stratified_group_median(result, df, group_cols)
            break  # once we fall back, all remaining are handled

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
section("1. LOAD DATA")

df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
print(f"  Raw shape: {df.shape}")
print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

# Drop rows with missing AOD or PM2.5 — these are the core variables
before = len(df)
df = df.dropna(subset=["AOD", "PM2.5"])
print(f"  Dropped {before - len(df)} rows missing AOD or PM2.5")
print(f"  Working rows: {len(df):,}")

GROUP_COLS = ["Monitoring_Station", "Season"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. REMOVE / FLAG PHYSICALLY IMPOSSIBLE VALUES
# ─────────────────────────────────────────────────────────────────────────────
section("2. PHYSICAL OUTLIER REMOVAL")

# BP < 900 hPa — confirmed sensor errors (normal Bangladesh BP ~995–1020 hPa)
bp_bad = df["BP"].notna() & (df["BP"] < 900)
print(f"  BP < 900 hPa (sensor errors): {bp_bad.sum()} rows → set to NaN")
df.loc[bp_bad, "BP"] = np.nan

# Temperature > 50°C — physically impossible for Bangladesh
temp_bad = df["Temperature"].notna() & (df["Temperature"] > 50)
print(f"  Temperature > 50°C: {temp_bad.sum()} rows → capped at 50")
df.loc[temp_bad, "Temperature"] = 50.0

# RH outside [0, 100]
rh_high = df["RH"].notna() & (df["RH"] > 100)
rh_low  = df["RH"].notna() & (df["RH"] < 0)
print(f"  RH > 100%: {rh_high.sum()} rows → capped at 100")
print(f"  RH < 0%:   {rh_low.sum()} rows → floored at 0")
df.loc[rh_high, "RH"] = 100.0
df.loc[rh_low,  "RH"] = 0.0

# Wind Speed > 50 m/s
ws_bad = df["Wind Speed"].notna() & (df["Wind Speed"] > 50)
print(f"  Wind Speed > 50 m/s: {ws_bad.sum()} rows → capped at 50")
df.loc[ws_bad, "Wind Speed"] = 50.0

# Rain extreme outlier cap (1613 mm single day is unrealistic for daily avg)
rain_p99 = df["Rain"].quantile(0.99)
rain_cap = rain_p99 * 3
rain_bad = df["Rain"].notna() & (df["Rain"] > rain_cap)
print(f"  Rain > {rain_cap:.1f} mm (3×p99): {rain_bad.sum()} rows → capped")
df.loc[rain_bad, "Rain"] = rain_cap

# ─────────────────────────────────────────────────────────────────────────────
# 3. DROP UNUSABLE COLUMN
# ─────────────────────────────────────────────────────────────────────────────
section("3. DROP UNUSABLE COLUMN")

# V Wind Speed: 95.4% missing — no reliable imputation possible
if "V Wind Speed" in df.columns:
    df.drop(columns=["V Wind Speed"], inplace=True)
    print("  Dropped 'V Wind Speed' (95.4% missing)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. STRATEGY A — Stratified group median
#    Temperature, RH, BP
# ─────────────────────────────────────────────────────────────────────────────
section("4. IMPUTATION — Strategy A: Stratified group median (Temp, RH, BP)")

for col in ["Temperature", "RH", "BP"]:
    n_miss = df[col].isna().sum()
    if n_miss == 0:
        print(f"  {col}: no missing values")
        continue
    df[col] = stratified_group_median(df[col], df, GROUP_COLS)
    remaining = df[col].isna().sum()
    print(f"  {col}: {n_miss:,} missing → filled. Remaining NaN: {remaining}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. STRATEGY B — KNN within group
#    Solar Rad, Wind Speed
# ─────────────────────────────────────────────────────────────────────────────
section("5. IMPUTATION — Strategy B: KNN within Station×Season (Solar Rad, Wind Speed)")

# KNN predictors: variables that are now fully imputed + Month
KNN_FEATURES = ["Temperature", "RH", "Rain", "Month"]
df["Month"] = df["Date"].dt.month    # needed for KNN

# Fill Rain first (only 94 missing, all Winter → 0.0)
n_rain_miss = df["Rain"].isna().sum()
df["Rain"] = df["Rain"].fillna(0.0)
print(f"  Rain: {n_rain_miss} missing → filled with 0.0 (all Winter, dry season)")

for col in ["Solar Rad", "Wind Speed"]:
    n_miss = df[col].isna().sum()
    if n_miss == 0:
        print(f"  {col}: no missing values")
        continue
    print(f"  {col}: {n_miss:,} missing → running KNN imputation...")
    df[col] = knn_within_group(df, col, KNN_FEATURES, GROUP_COLS, k=KNN_K)
    remaining = df[col].isna().sum()
    print(f"  {col}: done. Remaining NaN: {remaining}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. STRATEGY C — Wind_Origin → binary + missing flag
# ─────────────────────────────────────────────────────────────────────────────
section("6. IMPUTATION — Strategy C: Wind_Origin (binary + missing flag)")

n_wind_miss = df["Wind_Origin"].isna().sum()
df["Wind_Polluted"] = np.where(
    df["Wind_Origin"] == "Continental (Polluted)", 1.0,
    np.where(df["Wind_Origin"] == "Marine (Clean)", 0.0, np.nan)
)
df["Wind_Origin_missing"] = df["Wind_Polluted"].isna().astype(float)
df["Wind_Polluted"] = df["Wind_Polluted"].fillna(0.5)

print(f"  Wind_Origin: {n_wind_miss:,} missing")
print(f"  → Wind_Polluted:        Continental=1.0, Marine=0.0, Unknown=0.5")
print(f"  → Wind_Origin_missing:  1 where origin was unknown")

# ─────────────────────────────────────────────────────────────────────────────
# 7. WIND DIRECTION → cyclical + missing flag
# ─────────────────────────────────────────────────────────────────────────────
section("7. Wind Dir → sin/cos cyclical encoding + missing flag")

if "Wind Dir" in df.columns:
    n_wd_miss = df["Wind Dir"].isna().sum()
    df["WindDir_sin"]     = np.sin(np.deg2rad(df["Wind Dir"])).fillna(0.0)
    df["WindDir_cos"]     = np.cos(np.deg2rad(df["Wind Dir"])).fillna(0.0)
    df["WindDir_missing"] = df["Wind Dir"].isna().astype(float)
    df.drop(columns=["Wind Dir"], inplace=True)
    print(f"  Wind Dir: {n_wd_miss:,} missing → sin/cos=0.0, flag=1 where missing")

# ─────────────────────────────────────────────────────────────────────────────
# 8. ENCODE ALL REMAINING TEXT COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
section("8. ENCODE TEXT COLUMNS → numeric")

# Season (ordinal: pollution load order — Winter=most polluted)
season_map = {"Monsoon": 1, "Post-Monsoon": 2, "Pre-Monsoon": 3, "Winter": 4}
df["Season_ord"] = df["Season"].map(season_map)
df["Season_ord"] = stratified_group_median(df["Season_ord"], df, GROUP_COLS)
print(f"  Season → Season_ord  {season_map}")

# AOD_Loading (ordinal: magnitude order)
aod_map = {
    "Low AOD (<0.3)": 1,
    "Moderate AOD (0.3-0.8)": 2,
    "High AOD (>0.8)": 3
}
df["AOD_Loading_ord"] = df["AOD_Loading"].map(aod_map).fillna(2.0)
print(f"  AOD_Loading → AOD_Loading_ord  {aod_map}")

# Rain_Status (ordinal: intensity order)
rain_status_map = {"No Rain": 0, "Light Rain": 1, "Heavy Rain": 2}
df["Rain_Status_ord"] = df["Rain_Status"].map(rain_status_map).fillna(0.0)
print(f"  Rain_Status → Rain_Status_ord  {rain_status_map}")

# Humidity_Profile (ordinal: RH level)
humidity_map = {"Dry (<50%)": 1, "Moderate (50-75%)": 2, "Humid (>75%)": 3}
df["Humidity_ord"] = df["Humidity_Profile"].map(humidity_map)
df["Humidity_ord"] = stratified_group_median(df["Humidity_ord"], df, GROUP_COLS)
print(f"  Humidity_Profile → Humidity_ord  {humidity_map}")

# Temp_Profile (ordinal: temperature level)
temp_prof_map = {"Cool (<20°C)": 1, "Warm (20-28°C)": 2, "Hot (>28°C)": 3}
df["Temp_ord"] = df["Temp_Profile"].map(temp_prof_map)
df["Temp_ord"] = stratified_group_median(df["Temp_ord"], df, GROUP_COLS)
print(f"  Temp_Profile → Temp_ord  {temp_prof_map}")

# Geo_Zone → target encoding (mean PM2.5 per zone)
geo_mean = df.groupby("Geo_Zone")["PM2.5"].mean()
df["GeoZone_enc"] = df["Geo_Zone"].map(geo_mean).round(4)
print(f"  Geo_Zone → GeoZone_enc (mean PM2.5): {geo_mean.round(1).to_dict()}")

# Monitoring_Station → target encoding (mean PM2.5 per station)
station_mean = df.groupby("Monitoring_Station")["PM2.5"].mean()
df["Station_enc"] = df["Monitoring_Station"].map(station_mean).round(4)
print(f"  Monitoring_Station → Station_enc (mean PM2.5 per station)")

# ─────────────────────────────────────────────────────────────────────────────
# 9. DATE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
section("9. DATE FEATURES")

df["DayOfYear"] = df["Date"].dt.dayofyear
df["Year"]      = df["Date"].dt.year
df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12).round(6)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12).round(6)
print("  Month, DayOfYear, Year, Month_sin, Month_cos")

# ─────────────────────────────────────────────────────────────────────────────
# 10. BUILD DATASET A — Cleaned only, NO feature engineering
# ─────────────────────────────────────────────────────────────────────────────
section("10. BUILD DATASET A — Cleaned only (no feature engineering)")

COLS_A = [
    "Date",
    # Target
    "PM2.5",
    # Primary predictor — raw, uncorrected
    "AOD",
    # Meteorology — imputed, cleaned
    "Temperature", "RH", "Wind Speed", "Solar Rad", "BP", "Rain",
    # Date
    "Month", "DayOfYear", "Year",
    # Encoded categoricals
    "Season_ord", "AOD_Loading_ord", "Rain_Status_ord",
    "Humidity_ord", "Temp_ord",
    "Wind_Polluted", "Wind_Origin_missing",
    "WindDir_sin", "WindDir_cos", "WindDir_missing",
    "GeoZone_enc", "Station_enc",
    # Coordinates
    "Latitude", "Longitude",
]
COLS_A  = [c for c in COLS_A if c in df.columns]
df_A    = df[COLS_A].copy()

# Final NaN check — should be zero
nan_count_A = df_A.drop(columns=["Date"]).isnull().sum().sum()
print(f"  Columns : {len(COLS_A)}")
print(f"  Shape   : {df_A.shape}")
print(f"  Total NaN (excl. Date): {nan_count_A}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. BUILD DATASET B — Cleaned + Physical Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
section("11. BUILD DATASET B — Cleaned + Physical Feature Engineering")

# AOD nonlinear transforms
df["AOD_log"]  = np.log1p(df["AOD"].clip(lower=0))
df["AOD_sqrt"] = np.sqrt(df["AOD"].clip(lower=0))
df["AOD_sq"]   = df["AOD"] ** 2

# Boundary Layer proxy (Temp / Wind Speed)
df["BL_proxy"] = df["Temperature"] / (df["Wind Speed"] + 0.1)

# Hygroscopic growth factor f(RH) — Levy et al. 2007
df["f_RH"] = (1 - df["RH"] / 100).clip(lower=0.01) ** GAMMA_RH

# Corrected AOD features
df["AOD_BLH_corr"]  = df["AOD"] / df["BL_proxy"]
df["AOD_RH_corr"]   = df["AOD"] * df["f_RH"]
df["AOD_FULL_corr"] = df["AOD"] * df["f_RH"] / df["BL_proxy"]

# Physical interaction terms
df["AOD_x_RH"]      = df["AOD"] * df["RH"]
df["AOD_x_Temp"]    = df["AOD"] * df["Temperature"]
df["AOD_x_WS"]      = df["AOD"] * df["Wind Speed"]
df["AOD_x_SolRad"]  = df["AOD"] * df["Solar Rad"]
df["AOD_x_Season"]  = df["AOD"] * df["Season_ord"]
df["AOD_x_WindPol"] = df["AOD"] * df["Wind_Polluted"]
df["Wet_scavenge"]  = df["AOD"] * df["Rain"]

# Cyclical month encoding
df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

print("  AOD transforms      : AOD_log, AOD_sqrt, AOD_sq")
print("  BL correction       : BL_proxy = Temp / (WS + 0.1)")
print("  RH correction       : f_RH = (1 - RH/100)^0.5")
print("  Corrected AOD       : AOD_BLH_corr, AOD_RH_corr, AOD_FULL_corr")
print("  Interactions        : AOD×RH, AOD×Temp, AOD×WS, AOD×SolRad,")
print("                        AOD×Season, AOD×WindPolluted, Wet_scavenge")

COLS_B = [
    "Date",
    "PM2.5",
    "AOD", "AOD_log", "AOD_sqrt", "AOD_sq",
    "AOD_BLH_corr", "AOD_RH_corr", "AOD_FULL_corr",
    "BL_proxy", "f_RH",
    "Temperature", "RH", "Wind Speed", "Solar Rad", "BP", "Rain",
    "Wet_scavenge",
    "AOD_x_RH", "AOD_x_Temp", "AOD_x_WS", "AOD_x_SolRad",
    "AOD_x_Season", "AOD_x_WindPol",
    "Month", "Month_sin", "Month_cos", "DayOfYear", "Year",
    "Season_ord", "AOD_Loading_ord", "Rain_Status_ord",
    "Humidity_ord", "Temp_ord",
    "Wind_Polluted", "Wind_Origin_missing",
    "WindDir_sin", "WindDir_cos", "WindDir_missing",
    "GeoZone_enc", "Station_enc",
    "Latitude", "Longitude",
]
COLS_B  = [c for c in COLS_B if c in df.columns]
df_B    = df[COLS_B].copy()

nan_count_B = df_B.drop(columns=["Date"]).isnull().sum().sum()
print(f"\n  Columns : {len(COLS_B)}")
print(f"  Shape   : {df_B.shape}")
print(f"  Total NaN (excl. Date): {nan_count_B}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE DATASETS
# ─────────────────────────────────────────────────────────────────────────────
section("12. SAVE DATASETS")

path_A = os.path.join(SCRIPT_DIR, "Cleaned_Dataset_A.csv")
path_B = os.path.join(SCRIPT_DIR, "Cleaned_Dataset_B.csv")
df_A.to_csv(path_A, index=False)
df_B.to_csv(path_B, index=False)
print(f"  Saved Dataset A → {path_A}  ({df_A.shape[0]:,} × {df_A.shape[1]})")
print(f"  Saved Dataset B → {path_B}  ({df_B.shape[0]:,} × {df_B.shape[1]})")

# ─────────────────────────────────────────────────────────────────────────────
# 13. CORRELATION ANALYSIS ON DATASET A
#     (cleaned data, no engineering — honest baseline)
# ─────────────────────────────────────────────────────────────────────────────
section("13. CORRELATION ANALYSIS — Dataset A (cleaned, no engineering)")

# Feature columns to include in correlation analysis (exclude Date)
CORR_COLS_A = [c for c in COLS_A if c != "Date"]
data_corr = df_A[CORR_COLS_A].copy()

# ── Full Spearman matrix ────────────────────────────────────────────────────
spearman_matrix = data_corr.corr(method="spearman")
pearson_matrix  = data_corr.corr(method="pearson")

spearman_matrix.to_csv(
    os.path.join(SCRIPT_DIR, "Correlation_Matrix_Spearman.csv"))
pearson_matrix.to_csv(
    os.path.join(SCRIPT_DIR, "Correlation_Matrix_Pearson.csv"))
print("  Saved Correlation_Matrix_Spearman.csv")
print("  Saved Correlation_Matrix_Pearson.csv")

# ── Individual correlations with PM2.5 and AOD ─────────────────────────────
corr_rows = []
for col in CORR_COLS_A:
    if col in ["PM2.5", "AOD"]:
        continue
    for target in ["PM2.5", "AOD"]:
        valid = data_corr[[col, target]].dropna()
        if len(valid) < 20:
            continue
        sp_rho, sp_p = stats.spearmanr(valid[col], valid[target])
        pe_r,   pe_p = stats.pearsonr(valid[col],  valid[target])
        corr_rows.append({
            "Feature"           : col,
            "Target"            : target,
            "N"                 : len(valid),
            "Spearman_rho"      : round(sp_rho, 4),
            "Spearman_p"        : round(sp_p,   6),
            "Spearman_sig"      : "***" if sp_p < 0.001 else
                                  "**"  if sp_p < 0.01  else
                                  "*"   if sp_p < 0.05  else "ns",
            "Pearson_r"         : round(pe_r,   4),
            "Pearson_p"         : round(pe_p,   6),
            "Pearson_sig"       : "***" if pe_p < 0.001 else
                                  "**"  if pe_p < 0.01  else
                                  "*"   if pe_p < 0.05  else "ns",
        })

corr_df = pd.DataFrame(corr_rows)
corr_csv = os.path.join(SCRIPT_DIR, "Correlation_Results.csv")
corr_df.to_csv(corr_csv, index=False)
print(f"  Saved Correlation_Results.csv  ({len(corr_df)} rows)")

print("\n  Correlations with PM2.5:")
pm25_corr = (corr_df[corr_df["Target"] == "PM2.5"]
             .sort_values("Spearman_rho", key=abs, ascending=False))
print(pm25_corr[["Feature", "N", "Spearman_rho", "Spearman_sig",
                  "Pearson_r", "Pearson_sig"]].to_string(index=False))

print("\n  Correlations with AOD:")
aod_corr = (corr_df[corr_df["Target"] == "AOD"]
            .sort_values("Spearman_rho", key=abs, ascending=False))
print(aod_corr[["Feature", "N", "Spearman_rho", "Spearman_sig",
                 "Pearson_r", "Pearson_sig"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════
section("14. CORRELATION PLOTS")

# ─────────────────────────────────────────────────────────────────────────
# PLOT 1 — Full Spearman heatmap
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))
cmap_div = sns.diverging_palette(220, 20, as_cmap=True)
mask = np.triu(np.ones_like(spearman_matrix, dtype=bool), k=1)
sns.heatmap(
    spearman_matrix,
    annot=True, fmt=".2f",
    cmap=cmap_div, center=0, vmin=-1, vmax=1,
    ax=ax, linewidths=0.4,
    annot_kws={"size": 7},
    cbar_kws={"label": "Spearman ρ", "shrink": 0.8}
)
ax.set_title("Full Spearman Correlation Matrix — Dataset A (cleaned only)",
             fontsize=12, fontweight="bold", pad=12)
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
save_fig(fig, "01_spearman_full_matrix.png")

# ─────────────────────────────────────────────────────────────────────────
# PLOT 2 — Full Pearson heatmap
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    pearson_matrix,
    annot=True, fmt=".2f",
    cmap=cmap_div, center=0, vmin=-1, vmax=1,
    ax=ax, linewidths=0.4,
    annot_kws={"size": 7},
    cbar_kws={"label": "Pearson r", "shrink": 0.8}
)
ax.set_title("Full Pearson Correlation Matrix — Dataset A (cleaned only)",
             fontsize=12, fontweight="bold", pad=12)
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
save_fig(fig, "02_pearson_full_matrix.png")

# ─────────────────────────────────────────────────────────────────────────
# PLOT 3 — Spearman bar chart: all features vs PM2.5
# ─────────────────────────────────────────────────────────────────────────
pm25_sorted = pm25_corr.sort_values("Spearman_rho", ascending=True)
colors_bar  = ["#d73027" if v < 0 else "#2166ac"
               for v in pm25_sorted["Spearman_rho"]]

fig, ax = plt.subplots(figsize=(10, max(6, len(pm25_sorted) * 0.35)))
bars = ax.barh(pm25_sorted["Feature"], pm25_sorted["Spearman_rho"],
               color=colors_bar, edgecolor="white", linewidth=0.4)
ax.axvline(0, color="black", linewidth=0.8)
ax.axvline( 0.5, color="#2166ac", linestyle="--", linewidth=0.8, alpha=0.5,
            label="|ρ| = 0.5 (strong)")
ax.axvline(-0.5, color="#2166ac", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axvline( 0.35, color="#74add1", linestyle=":", linewidth=0.8, alpha=0.5,
            label="|ρ| = 0.35 (moderate)")
ax.axvline(-0.35, color="#74add1", linestyle=":", linewidth=0.8, alpha=0.5)

for bar, (_, row) in zip(bars, pm25_sorted.iterrows()):
    xpos = row["Spearman_rho"]
    ha   = "left" if xpos >= 0 else "right"
    offset = 0.005 if xpos >= 0 else -0.005
    ax.text(xpos + offset, bar.get_y() + bar.get_height() / 2,
            f"{xpos:+.3f}{row['Spearman_sig']}",
            va="center", ha=ha, fontsize=8)

ax.set_xlabel("Spearman ρ", fontsize=10)
ax.set_title("Spearman Correlation with PM2.5\n(Dataset A — cleaned, no feature engineering)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(
    pm25_sorted["Spearman_rho"].min() - 0.12,
    pm25_sorted["Spearman_rho"].max() + 0.12
)
plt.tight_layout()
save_fig(fig, "03_spearman_vs_PM25.png")

# ─────────────────────────────────────────────────────────────────────────
# PLOT 4 — Spearman bar chart: all features vs AOD
# ─────────────────────────────────────────────────────────────────────────
aod_sorted = aod_corr.sort_values("Spearman_rho", ascending=True)
colors_aod = ["#d73027" if v < 0 else "#1a9641"
              for v in aod_sorted["Spearman_rho"]]

fig, ax = plt.subplots(figsize=(10, max(6, len(aod_sorted) * 0.35)))
bars = ax.barh(aod_sorted["Feature"], aod_sorted["Spearman_rho"],
               color=colors_aod, edgecolor="white", linewidth=0.4)
ax.axvline(0, color="black", linewidth=0.8)
ax.axvline( 0.5, color="#1a9641", linestyle="--", linewidth=0.8, alpha=0.5,
            label="|ρ| = 0.5 (strong)")
ax.axvline(-0.5, color="#1a9641", linestyle="--", linewidth=0.8, alpha=0.5)

for bar, (_, row) in zip(bars, aod_sorted.iterrows()):
    xpos = row["Spearman_rho"]
    ha   = "left" if xpos >= 0 else "right"
    offset = 0.003 if xpos >= 0 else -0.003
    ax.text(xpos + offset, bar.get_y() + bar.get_height() / 2,
            f"{xpos:+.3f}{row['Spearman_sig']}",
            va="center", ha=ha, fontsize=8)

ax.set_xlabel("Spearman ρ", fontsize=10)
ax.set_title("Spearman Correlation with AOD\n(Dataset A — cleaned, no feature engineering)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(
    aod_sorted["Spearman_rho"].min() - 0.10,
    aod_sorted["Spearman_rho"].max() + 0.10
)
plt.tight_layout()
save_fig(fig, "04_spearman_vs_AOD.png")

# ─────────────────────────────────────────────────────────────────────────
# PLOT 5 — Spearman vs Pearson comparison for PM2.5 correlations
# ─────────────────────────────────────────────────────────────────────────
pm25_comp = (corr_df[corr_df["Target"] == "PM2.5"]
             .sort_values("Spearman_rho", key=abs, ascending=False)
             .head(15))

x  = np.arange(len(pm25_comp))
w  = 0.38
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - w/2, pm25_comp["Spearman_rho"], w,
       label="Spearman ρ", color="#2166ac", alpha=0.85, edgecolor="white")
ax.bar(x + w/2, pm25_comp["Pearson_r"],    w,
       label="Pearson r",  color="#d73027", alpha=0.85, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.7)
ax.axhline( 0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
ax.axhline(-0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(pm25_comp["Feature"], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Correlation coefficient")
ax.set_title("Spearman vs Pearson — top 15 features correlated with PM2.5\n"
             "(gap between bars = nonlinearity in that relationship)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
save_fig(fig, "05_spearman_vs_pearson_PM25.png")

# ─────────────────────────────────────────────────────────────────────────
# PLOT 6 — AOD vs PM2.5 scatter by season with ρ annotation
# ─────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("AOD vs PM2.5 by Season — Dataset A (cleaned, no engineering)\n"
             "Spearman ρ and Pearson r shown per panel",
             fontsize=11, fontweight="bold")

for ax, season in zip(axes.flat, SEASON_ORDER):
    sub = df_A[df_A["Season_ord"] ==
               {"Winter": 4, "Pre-Monsoon": 3,
                "Monsoon": 1, "Post-Monsoon": 2}[season]].dropna(
               subset=["AOD", "PM2.5"])

    ax.scatter(sub["AOD"], sub["PM2.5"],
               alpha=0.25, s=9, color=SEASON_COLORS[season])

    sp_rho, sp_p = stats.spearmanr(sub["AOD"], sub["PM2.5"])
    pe_r,   pe_p = stats.pearsonr(sub["AOD"],  sub["PM2.5"])
    sp_sig = "***" if sp_p < 0.001 else "**" if sp_p < 0.01 else "*" if sp_p < 0.05 else "ns"
    pe_sig = "***" if pe_p < 0.001 else "**" if pe_p < 0.01 else "*" if pe_p < 0.05 else "ns"

    # Trend line
    if len(sub) > 10:
        z = np.polyfit(sub["AOD"], sub["PM2.5"], 1)
        xline = np.linspace(sub["AOD"].min(), sub["AOD"].max(), 100)
        ax.plot(xline, np.poly1d(z)(xline),
                "k--", linewidth=1.2, alpha=0.6, label="Linear trend")

    ax.set_title(
        f"{season}  (n={len(sub):,})\n"
        f"Spearman ρ={sp_rho:+.3f}{sp_sig}   Pearson r={pe_r:+.3f}{pe_sig}",
        fontsize=9.5, color=SEASON_COLORS[season], fontweight="bold")
    ax.set_xlabel("AOD")
    ax.set_ylabel("PM2.5 (µg/m³)")

plt.tight_layout()
save_fig(fig, "06_AOD_PM25_scatter_by_season.png")

# ─────────────────────────────────────────────────────────────────────────
# PLOT 7 — Correlation with PM2.5 broken down by season
#          (shows how relationship changes across seasons)
# ─────────────────────────────────────────────────────────────────────────
season_num_map = {"Monsoon": 1, "Post-Monsoon": 2, "Pre-Monsoon": 3, "Winter": 4}
base_met_vars  = [c for c in
                  ["AOD", "Temperature", "RH", "Wind Speed",
                   "Solar Rad", "BP", "Rain"]
                  if c in df_A.columns]

season_corr_rows = []
for season in SEASON_ORDER:
    snum = season_num_map[season]
    sub  = df_A[df_A["Season_ord"] == snum]
    for var in base_met_vars:
        valid = sub[["PM2.5", var]].dropna()
        if len(valid) < 20:
            continue
        rho, p = stats.spearmanr(valid["PM2.5"], valid[var])
        season_corr_rows.append({
            "Season": season, "Feature": var,
            "Spearman_rho": round(rho, 4), "p_value": round(p, 6),
            "N": len(valid)
        })

season_corr_df = pd.DataFrame(season_corr_rows)
pivot_season   = season_corr_df.pivot(
    index="Feature", columns="Season", values="Spearman_rho")
pivot_season   = pivot_season.reindex(columns=SEASON_ORDER)

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(
    pivot_season,
    annot=True, fmt=".3f",
    cmap=sns.diverging_palette(220, 20, as_cmap=True),
    center=0, vmin=-1, vmax=1, ax=ax,
    linewidths=0.5, annot_kws={"size": 10},
    cbar_kws={"label": "Spearman ρ with PM2.5", "shrink": 0.8}
)
ax.set_title("Spearman ρ with PM2.5 — per season\n"
             "(reveals which features are useful in each meteorological regime)",
             fontsize=10, fontweight="bold")
ax.set_xlabel("Season")
ax.set_ylabel("Feature")
plt.tight_layout()
save_fig(fig, "07_spearman_PM25_by_season_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────
# PLOT 8 — Dataset A vs B: Spearman rho of AOD variants with PM2.5
#          (key plot proving corrections improve the AOD signal)
# ─────────────────────────────────────────────────────────────────────────
aod_variants = {
    "AOD (raw)"       : "AOD",
    "AOD_log"         : "AOD_log",
    "AOD_sqrt"        : "AOD_sqrt",
    "AOD_sq"          : "AOD_sq",
    "AOD_BLH_corr"    : "AOD_BLH_corr",
    "AOD_RH_corr"     : "AOD_RH_corr",
    "AOD_FULL_corr"   : "AOD_FULL_corr",
}

variant_rows = []
for label, col in aod_variants.items():
    if col not in df_B.columns:
        continue
    valid = df_B[["PM2.5", col]].dropna()
    if len(valid) < 20:
        continue
    sp_rho, sp_p = stats.spearmanr(valid["PM2.5"], valid[col])
    pe_r,   pe_p = stats.pearsonr(valid["PM2.5"],  valid[col])
    variant_rows.append({
        "Label"       : label,
        "Spearman_rho": round(sp_rho, 4),
        "Spearman_p"  : round(sp_p,   6),
        "Pearson_r"   : round(pe_r,   4),
        "N"           : len(valid),
        "Corrected"   : "Yes" if "corr" in col else "No"
    })

variant_df = pd.DataFrame(variant_rows)

x     = np.arange(len(variant_df))
w     = 0.38
colors_v = ["#d73027" if r["Corrected"] == "No" else "#1a9641"
            for _, r in variant_df.iterrows()]

fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - w/2, variant_df["Spearman_rho"], w,
            color=colors_v, alpha=0.85, edgecolor="white", label="Spearman ρ")
ax.bar(x + w/2, variant_df["Pearson_r"], w,
       color=colors_v, alpha=0.45, edgecolor="white",
       hatch="///", label="Pearson r")
ax.axhline(0, color="black", linewidth=0.7)
ax.axhline(0.5,  color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(-0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

for bar, (_, row) in zip(b1, variant_df.iterrows()):
    ypos = row["Spearman_rho"]
    va   = "bottom" if ypos >= 0 else "top"
    offset = 0.005 if ypos >= 0 else -0.005
    ax.text(bar.get_x() + bar.get_width() / 2, ypos + offset,
            f"{ypos:+.3f}", ha="center", va=va, fontsize=8.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(variant_df["Label"], rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Correlation with PM2.5")
ax.set_title("Raw AOD vs physically corrected AOD variants — correlation with PM2.5\n"
             "Red = no correction  |  Green = physically corrected  |  "
             "Solid = Spearman  |  Hatched = Pearson",
             fontsize=10, fontweight="bold")

raw_patch  = plt.Rectangle((0,0),1,1, fc="#d73027", alpha=0.85, label="No correction")
corr_patch = plt.Rectangle((0,0),1,1, fc="#1a9641", alpha=0.85, label="Corrected")
ax.legend(handles=[raw_patch, corr_patch], fontsize=9, loc="upper right")
plt.tight_layout()
save_fig(fig, "08_AOD_variants_correlation_PM25.png")

# ─────────────────────────────────────────────────────────────────────────
# PLOT 9 — Scatter matrix: core variables coloured by season
# ─────────────────────────────────────────────────────────────────────────
scatter_vars = [c for c in
                ["PM2.5", "AOD", "Temperature", "RH", "Wind Speed", "BP"]
                if c in df_A.columns]
df_scatter   = df_A[scatter_vars + ["Season_ord"]].dropna()
df_scatter["Season_label"] = df_scatter["Season_ord"].map(
    {v: k for k, v in season_num_map.items()})

fig = plt.figure(figsize=(14, 12))
n   = len(scatter_vars)
for i, v1 in enumerate(scatter_vars):
    for j, v2 in enumerate(scatter_vars):
        ax = fig.add_subplot(n, n, i * n + j + 1)
        if i == j:
            for season, color in SEASON_COLORS.items():
                sub = df_scatter[df_scatter["Season_label"] == season]
                ax.hist(sub[v1].dropna(), bins=30, alpha=0.5,
                        color=color, density=True)
            ax.set_xlabel(v1, fontsize=7)
        else:
            for season, color in SEASON_COLORS.items():
                sub = df_scatter[df_scatter["Season_label"] == season]
                ax.scatter(sub[v2], sub[v1],
                           alpha=0.15, s=4, color=color)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel(v1, fontsize=7)
        if i == n - 1:
            ax.set_xlabel(v2, fontsize=7)

# Legend
handles = [plt.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=c, markersize=7, label=s)
           for s, c in SEASON_COLORS.items()]
fig.legend(handles=handles, loc="upper right",
           fontsize=8, title="Season", framealpha=0.9)
fig.suptitle("Scatter matrix — core variables coloured by season\n"
             "(diagonal = density distribution per season)",
             fontsize=11, fontweight="bold", y=1.002)
plt.tight_layout()
save_fig(fig, "09_scatter_matrix_by_season.png")

# ─────────────────────────────────────────────────────────────────────────
# PLOT 10 — AOD vs PM2.5 with regression per Geo_Zone
# ─────────────────────────────────────────────────────────────────────────
GEO_COLORS = {
    "Coastal"          : "#1f78b4",
    "Inland_Urban"     : "#e31a1c",
    "Inland_SemiUrban" : "#33a02c",
}
if "Geo_Zone" in df.columns:
    fig, ax = plt.subplots(figsize=(9, 6))
    for gz, color in GEO_COLORS.items():
        sub = df[df["Geo_Zone"] == gz].dropna(subset=["AOD", "PM2.5"])
        ax.scatter(sub["AOD"], sub["PM2.5"],
                   alpha=0.2, s=8, color=color, label=gz)
        if len(sub) > 10:
            z = np.polyfit(sub["AOD"], sub["PM2.5"], 1)
            xline = np.linspace(sub["AOD"].min(), sub["AOD"].max(), 100)
            ax.plot(xline, np.poly1d(z)(xline),
                    color=color, linewidth=2, alpha=0.9)
        rho, p = stats.spearmanr(sub["AOD"], sub["PM2.5"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {gz:22s}: ρ={rho:+.3f}{sig}  n={len(sub):,}")

    ax.set_xlabel("AOD")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("AOD vs PM2.5 by Geo Zone with regression lines\n"
                 "(Spearman ρ printed to console)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig, "10_AOD_PM25_by_geo_zone.png")

# ─────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────
section("SUMMARY")
print(f"  Dataset A  : {df_A.shape[0]:,} rows × {df_A.shape[1]} cols  → Cleaned_Dataset_A.csv")
print(f"  Dataset B  : {df_B.shape[0]:,} rows × {df_B.shape[1]} cols  → Cleaned_Dataset_B.csv")
print(f"  Corr table : {len(corr_df)} rows                → Correlation_Results.csv")
print(f"  Spearman M : {spearman_matrix.shape}             → Correlation_Matrix_Spearman.csv")
print(f"  Pearson  M : {pearson_matrix.shape}              → Correlation_Matrix_Pearson.csv")
print(f"  Plots      : {len(os.listdir(PLOT_DIR))} files   → {PLOT_DIR}/")
print("\n  Done.")