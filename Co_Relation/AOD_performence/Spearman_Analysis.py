"""
=============================================================================
FEATURE ENGINEERING — AOD Proxy Study
Bangladesh Dataset

WHAT THIS SCRIPT DOES (one job only):
  Reads the raw CSV and produces two clean, fully numeric datasets:

  Dataset_A_Raw.csv
      AOD (unchanged) + meteorological variables (imputed) + encoded
      categoricals. No physics corrections. This is the "naive" dataset
      the first ML model will be trained on.

  Dataset_B_Corrected.csv
      Everything in A, PLUS physically motivated correction features:
        - BL_proxy        : boundary layer height approximation
        - f_RH            : hygroscopic growth factor
        - AOD_BLH_corr    : AOD normalised by boundary layer proxy
        - AOD_RH_corr     : AOD dampened by hygroscopic factor
        - AOD_FULL_corr   : combined BLH + RH correction (literature Eq.)
        - wet_scavenge    : Rain × AOD interaction
        - AOD nonlinear transforms (log, sqrt, square)
        - Cyclical time   : Month sin/cos  (Dec ≈ Jan continuity)
        - Location means  : station and geo-zone PM2.5 mean (target encode)

  After running this script, feed both CSVs to your ML training script.
  The improvement in R² from A → B proves that raw AOD alone is insufficient.

ENCODING DECISIONS (all text removed, every column is numeric):
  Season          → ordinal int  (Monsoon=1 … Winter=4, pollution load order)
  AOD_Loading     → ordinal int  (Low=1, Moderate=2, High=3)
  Rain_Status     → ordinal int  (No Rain=0, Light=1, Heavy=2)
  Humidity_Profile→ ordinal int  (Dry=1, Moderate=2, Humid=3)
  Temp_Profile    → ordinal int  (Cool=1, Warm=2, Hot=3)
  Wind_Origin     → binary float (Continental=1.0, Marine=0.0, missing=0.5)
  Geo_Zone        → target-encoded float (mean PM2.5 per zone)
  Monitoring_Station → target-encoded float (mean PM2.5 per station)
  Date            → Month (int) + DayOfYear (int) + Month_sin + Month_cos

PHYSICAL CORRECTION FORMULAE (Method 1, from literature):
  BL_proxy      = Temperature / (Wind_Speed + 0.1)
                  proxy for boundary layer height — higher T + calm wind
                  = shallower BL = higher surface PM2.5 concentration
  f_RH          = (1 - RH/100) ^ 0.5
                  hygroscopic growth dampening factor (Levy et al. 2007)
                  removes the optical inflation AOD gets from humidity
  AOD_BLH_corr  = AOD / BL_proxy
                  normalises AOD by mixing layer depth
  AOD_RH_corr   = AOD * f_RH
                  removes hygroscopic AOD inflation
  AOD_FULL_corr = AOD * f_RH / BL_proxy
                  full correction: AOD/(BLH * f(RH)) — the gold standard
                  from van Donkelaar et al. 2010 and Hossen & Frey 2018

REQUIREMENTS:
  pip install pandas numpy

USAGE:
  python feature_engineering.py
  (run in the same folder as Master_Dataset_Final_QC.csv)
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FILE  = "Master_Dataset_Final_QC.csv"
OUTPUT_A    = "Dataset_A_Raw.csv"
OUTPUT_B    = "Dataset_B_Corrected.csv"
OUTPUT_LOG  = "feature_engineering_log.txt"

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING HELPER
# ─────────────────────────────────────────────────────────────────────────────
log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(msg)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 70)
log("  FEATURE ENGINEERING PIPELINE")
log("  Bangladesh AOD–PM2.5 Proxy Study")
log("=" * 70)

df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
log(f"\n[LOAD] {len(df):,} rows  |  {df.shape[1]} columns")

# Keep only rows where target (PM2.5) and primary predictor (AOD) exist
before = len(df)
df = df.dropna(subset=["AOD", "PM2.5"])
log(f"[LOAD] Dropped {before - len(df)} rows missing AOD or PM2.5")
log(f"[LOAD] Working rows: {len(df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  REMOVE UNUSABLE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 1] Removing unusable columns")

# V Wind Speed: 95.4% missing — cannot be imputed reliably
# Wind Dir: 48% missing AND circular (would need sin/cos + huge imputation)
#           kept as WindDir_sin / WindDir_cos only where available
drop_cols = ["V Wind Speed"]
for c in drop_cols:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)
        log(f"  Dropped '{c}' (>95% missing)")

# Remove BP outliers: values < 900 hPa are sensor errors
bp_bad = (df["BP"] < 900) & df["BP"].notna()
log(f"  Removing {bp_bad.sum()} rows with BP < 900 hPa (sensor errors)")
df = df[~bp_bad].copy()
log(f"  Rows after BP cleanup: {len(df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  EXTRACT DATE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 2] Date → numeric features")

df["Month"]     = df["Date"].dt.month        # 1–12
df["DayOfYear"] = df["Date"].dt.dayofyear    # 1–365
df["Year"]      = df["Date"].dt.year

# Cyclical encoding: December and January are adjacent, not 11 apart
df["Month_sin"] = np.sin(2 * np.pi * df["Month"]     / 12).round(6)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"]     / 12).round(6)
df["DOY_sin"]   = np.sin(2 * np.pi * df["DayOfYear"] / 365).round(6)
df["DOY_cos"]   = np.cos(2 * np.pi * df["DayOfYear"] / 365).round(6)

log("  Month         → Month (int 1-12), Month_sin, Month_cos")
log("  DayOfYear     → DayOfYear (int), DOY_sin, DOY_cos")
log("  Year          → Year (int, captures inter-annual trend)")

# Wind direction cyclical (where available)
if "Wind Dir" in df.columns:
    df["WindDir_sin"] = np.sin(np.deg2rad(df["Wind Dir"])).round(6)
    df["WindDir_cos"] = np.cos(np.deg2rad(df["Wind Dir"])).round(6)
    # These will be NaN where Wind Dir is missing — that is fine
    df.drop(columns=["Wind Dir"], inplace=True)
    log("  Wind Dir      → WindDir_sin, WindDir_cos  (NaN where missing)")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  ENCODE ALL TEXT / CATEGORICAL COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 3] Encoding all categorical/text columns → numeric")

# ── 4a. Season (ordinal by pollution load) ───────────────────────────────────
# Monsoon=1 (cleanest air, wet scavenging, marine)
# Winter=4  (worst air, dry stable BL, continental aerosols)
# This ordering is physically motivated, not alphabetical
season_map = {
    "Monsoon"      : 1,
    "Post-Monsoon" : 2,
    "Pre-Monsoon"  : 3,
    "Winter"       : 4,
}
df["Season_ord"] = df["Season"].map(season_map)
missing_season = df["Season_ord"].isna().sum()
if missing_season > 0:
    df["Season_ord"] = df["Season_ord"].fillna(df["Season_ord"].median())
    log(f"  [WARN] {missing_season} rows had unknown Season → filled with median")
log(f"  Season → Season_ord  {season_map}")

# ── 4b. AOD_Loading (ordinal: natural AOD magnitude order) ───────────────────
aod_load_map = {
    "Low AOD (<0.3)"        : 1,
    "Moderate AOD (0.3-0.8)": 2,
    "High AOD (>0.8)"       : 3,
}
df["AOD_Loading_ord"] = df["AOD_Loading"].map(aod_load_map)
df["AOD_Loading_ord"] = df["AOD_Loading_ord"].fillna(
    df["AOD_Loading_ord"].median()
)
log(f"  AOD_Loading → AOD_Loading_ord  {aod_load_map}")

# ── 4c. Rain_Status (ordinal: intensity order) ───────────────────────────────
rain_map = {
    "No Rain"    : 0,
    "Light Rain" : 1,
    "Heavy Rain" : 2,
}
df["Rain_Status_ord"] = df["Rain_Status"].map(rain_map)
df["Rain_Status_ord"] = df["Rain_Status_ord"].fillna(0)  # assume no rain if unknown
log(f"  Rain_Status → Rain_Status_ord  {rain_map}")

# ── 4d. Humidity_Profile (ordinal: RH level order) ──────────────────────────
humidity_map = {
    "Dry (<50%)"       : 1,
    "Moderate (50-75%)": 2,
    "Humid (>75%)"     : 3,
}
df["Humidity_ord"] = df["Humidity_Profile"].map(humidity_map)
df["Humidity_ord"] = df["Humidity_ord"].fillna(df["Humidity_ord"].median())
log(f"  Humidity_Profile → Humidity_ord  {humidity_map}")

# ── 4e. Temp_Profile (ordinal: temperature level order) ─────────────────────
temp_map = {
    "Cool (<20\u00b0C)" : 1,
    "Warm (20-28\u00b0C)": 2,
    "Hot (>28\u00b0C)"  : 3,
}
df["Temp_ord"] = df["Temp_Profile"].map(temp_map)
df["Temp_ord"] = df["Temp_ord"].fillna(df["Temp_ord"].median())
log(f"  Temp_Profile → Temp_ord  {temp_map}")

# ── 4f. Wind_Origin (binary: physical interpretation) ───────────────────────
# Continental (Polluted) = 1.0  — high aerosol load, consistent composition
# Marine (Clean)         = 0.0  — low load, sea-salt dominated
# Missing                = 0.5  — uncertain (do not assume either)
df["Wind_Polluted"] = np.where(
    df["Wind_Origin"] == "Continental (Polluted)", 1.0,
    np.where(df["Wind_Origin"] == "Marine (Clean)", 0.0, 0.5)
)
log(f"  Wind_Origin → Wind_Polluted  (Continental=1.0, Marine=0.0, missing=0.5)")

# ── 4g. Geo_Zone (target encoding: mean PM2.5 per zone) ─────────────────────
# Target encoding captures the spatial pollution signal directly
geo_mean = df.groupby("Geo_Zone")["PM2.5"].mean()
df["GeoZone_enc"] = df["Geo_Zone"].map(geo_mean).round(4)
log(f"  Geo_Zone → GeoZone_enc (target: mean PM2.5)")
for k, v in geo_mean.items():
    log(f"    {k:25s} → {v:.2f} µg/m³")

# ── 4h. Monitoring_Station (target encoding: mean PM2.5 per station) ────────
station_mean = df.groupby("Monitoring_Station")["PM2.5"].mean()
df["Station_enc"] = df["Monitoring_Station"].map(station_mean).round(4)
log(f"  Monitoring_Station → Station_enc (target: mean PM2.5)")
for k, v in station_mean.items():
    log(f"    {k:30s} → {v:.2f} µg/m³")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  IMPUTE MISSING NUMERIC VALUES
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 4] Imputing missing numeric values with column median")

numeric_cols_to_fill = [
    "Wind Speed", "Temperature", "RH", "Solar Rad", "BP", "Rain",
    "WindDir_sin", "WindDir_cos"
]

for col in numeric_cols_to_fill:
    if col not in df.columns:
        continue
    n_missing = df[col].isna().sum()
    if n_missing > 0:
        med = df[col].median()
        df[col] = df[col].fillna(med)
        log(f"  {col:15s}: filled {n_missing:5,} missing → median={med:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SANITY-CHECK NUMERIC RANGES
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 5] Sanity checks on numeric ranges")

# Temperature: values > 50°C are suspect for Bangladesh
temp_high = (df["Temperature"] > 50).sum()
if temp_high > 0:
    log(f"  [WARN] {temp_high} Temperature values > 50°C — capping at 50°C")
    df["Temperature"] = df["Temperature"].clip(upper=50)

# RH: must be 0–100%
rh_high = (df["RH"] > 100).sum()
if rh_high > 0:
    log(f"  [WARN] {rh_high} RH values > 100% — capping at 100")
    df["RH"] = df["RH"].clip(upper=100)

rh_low = (df["RH"] < 0).sum()
if rh_low > 0:
    log(f"  [WARN] {rh_low} RH values < 0% — flooring at 0")
    df["RH"] = df["RH"].clip(lower=0)

# Wind Speed: 0–42 m/s found in data, 42 is extreme but not impossible
ws_high = (df["Wind Speed"] > 50).sum()
if ws_high > 0:
    log(f"  [WARN] {ws_high} Wind Speed values > 50 m/s — capping")
    df["Wind Speed"] = df["Wind Speed"].clip(upper=50)

# Rain: 1613 mm single day is extreme outlier
rain_p99 = df["Rain"].quantile(0.99)
rain_high = (df["Rain"] > rain_p99 * 3).sum()
if rain_high > 0:
    log(f"  [WARN] {rain_high} Rain values > 3×p99 ({rain_p99*3:.1f} mm) — capping")
    df["Rain"] = df["Rain"].clip(upper=rain_p99 * 3)

log("  Sanity checks complete")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  BUILD DATASET A  (raw — no physics corrections)
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 6] Building Dataset A — Raw (no physical corrections)")

cols_A = [
    "Date",
    # Target
    "PM2.5",
    # Primary predictor (raw, uncorrected)
    "AOD",
    # Raw meteorology (imputed, cleaned)
    "Wind Speed", "Temperature", "RH", "Solar Rad", "BP", "Rain",
    # Cyclical time
    "Month", "DayOfYear", "Year",
    "Month_sin", "Month_cos", "DOY_sin", "DOY_cos",
    # Cyclical wind direction
    "WindDir_sin", "WindDir_cos",
    # Encoded categoricals (fully numeric)
    "Season_ord", "AOD_Loading_ord", "Rain_Status_ord",
    "Humidity_ord", "Temp_ord", "Wind_Polluted",
    "GeoZone_enc", "Station_enc",
    # Coordinates (spatial signal)
    "Latitude", "Longitude",
]

# Keep only columns that actually exist in df
cols_A = [c for c in cols_A if c in df.columns]
df_A = df[cols_A].copy()

log(f"  Dataset A columns ({len(cols_A)}): {cols_A}")
log(f"  Dataset A shape   : {df_A.shape}")
log(f"  Any NaN in A      : {df_A.drop(columns=['Date']).isnull().any().any()}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  BUILD DATASET B  (corrected — physics-based AOD transformations)
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 7] Building Dataset B — Corrected (physical AOD corrections)")

# ── 8a. AOD nonlinear transforms ─────────────────────────────────────────────
# The AOD–PM2.5 relationship is known to be nonlinear
# These allow tree models and linear models alike to fit nonlinear patterns
df["AOD_log"]  = np.log1p(df["AOD"].clip(lower=0)).round(6)   # log(1+AOD)
df["AOD_sqrt"] = np.sqrt(df["AOD"].clip(lower=0)).round(6)    # sqrt(AOD)
df["AOD_sq"]   = (df["AOD"] ** 2).round(6)                     # AOD²
log("  AOD transforms: AOD_log = log1p(AOD), AOD_sqrt, AOD_sq")

# ── 8b. Boundary Layer Height proxy ──────────────────────────────────────────
# BLH is the dominant correction factor (Levy et al. 2007)
# Higher temperature + lower wind speed = shallower, more stable boundary layer
# → same PM2.5 emission concentrates near surface → higher surface PM2.5
# We don't have direct BLH so we approximate:
#   BL_proxy = Temperature / (Wind_Speed + 0.1)
# +0.1 prevents division by zero on calm days
df["BL_proxy"] = (df["Temperature"] / (df["Wind Speed"] + 0.1)).round(6)
log("  BL_proxy = Temperature / (Wind_Speed + 0.1)")
log(f"    BL_proxy stats: min={df['BL_proxy'].min():.2f}  "
    f"median={df['BL_proxy'].median():.2f}  max={df['BL_proxy'].max():.2f}")

# ── 8c. Hygroscopic growth factor f(RH) ──────────────────────────────────────
# AOD is inflated by humidity because aerosol particles absorb water
# and grow in size, scattering more light without increasing PM2.5 mass
# Literature formula (Levy et al. 2007, van Donkelaar et al. 2010):
#   f(RH) = (1 - RH/100) ^ gamma    where gamma ≈ 0.5 for South Asian aerosols
# f_RH close to 1 → low humidity → AOD not inflated → better proxy
# f_RH close to 0 → saturated air → AOD heavily inflated → poor proxy
GAMMA = 0.5
df["f_RH"] = ((1 - df["RH"] / 100).clip(lower=0.01) ** GAMMA).round(6)
log(f"  f_RH = (1 - RH/100)^{GAMMA}  (hygroscopic dampening factor)")
log(f"    f_RH=1.0 at RH=0% (no inflation), f_RH→0 at RH=100% (fully inflated)")
log(f"    f_RH stats: min={df['f_RH'].min():.4f}  "
    f"median={df['f_RH'].median():.4f}  max={df['f_RH'].max():.4f}")

# ── 8d. AOD corrected for boundary layer ─────────────────────────────────────
# AOD_BLH_corr: removes the vertical mismatch between column AOD and surface PM2.5
# Physically: PM2.5 ∝ AOD / BLH  →  high BLH means same AOD is diluted vertically
df["AOD_BLH_corr"] = (df["AOD"] / df["BL_proxy"]).round(6)
log("  AOD_BLH_corr = AOD / BL_proxy")

# ── 8e. AOD corrected for hygroscopic growth ─────────────────────────────────
# AOD_RH_corr: removes the optical inflation from humidity
df["AOD_RH_corr"] = (df["AOD"] * df["f_RH"]).round(6)
log("  AOD_RH_corr = AOD × f_RH  (removes humidity inflation)")

# ── 8f. FULL correction: both BLH and RH ─────────────────────────────────────
# This is the core Method 1 correction from the literature
# AOD_FULL_corr ∝  PM2.5  more closely than raw AOD
# This is what we expect to be the best proxy feature
df["AOD_FULL_corr"] = (df["AOD"] * df["f_RH"] / df["BL_proxy"]).round(6)
log("  AOD_FULL_corr = AOD × f_RH / BL_proxy  ← FULL correction (Method 1)")
log(f"    AOD_FULL_corr stats: min={df['AOD_FULL_corr'].min():.4f}  "
    f"median={df['AOD_FULL_corr'].median():.4f}  "
    f"max={df['AOD_FULL_corr'].max():.4f}")

# ── 8g. Wet scavenging interaction ───────────────────────────────────────────
# Rain physically removes PM2.5 from the surface layer (wet scavenging)
# but AOD may remain high above the rain cloud
# This interaction term captures that decoupling
df["Wet_scavenge"] = (df["Rain"] * df["AOD"]).round(6)
log("  Wet_scavenge = Rain × AOD  (captures rain-induced AOD/PM2.5 decoupling)")

# ── 8h. Physical interaction terms ───────────────────────────────────────────
# Each captures a known physical mechanism that modulates AOD→PM2.5
df["AOD_x_RH"]      = (df["AOD"] * df["RH"]).round(6)
df["AOD_x_Temp"]    = (df["AOD"] * df["Temperature"]).round(6)
df["AOD_x_WS"]      = (df["AOD"] * df["Wind Speed"]).round(6)
df["AOD_x_SolRad"]  = (df["AOD"] * df["Solar Rad"]).round(6)
df["AOD_x_Season"]  = (df["AOD"] * df["Season_ord"]).round(6)
df["AOD_x_WindPol"] = (df["AOD"] * df["Wind_Polluted"]).round(6)

log("  Physical interactions:")
log("    AOD_x_RH      = AOD × RH         (hygroscopic regime)")
log("    AOD_x_Temp    = AOD × Temperature (BL mixing regime)")
log("    AOD_x_WS      = AOD × Wind Speed  (dispersion regime)")
log("    AOD_x_SolRad  = AOD × Solar Rad   (photochemical regime)")
log("    AOD_x_Season  = AOD × Season_ord  (seasonal aerosol type)")
log("    AOD_x_WindPol = AOD × Wind_Polluted (source type regime)")

# ── 8i. Stability index: inverse of proxy ratio variability ──────────────────
# The proxy ratio PM2.5/AOD should be stable for AOD to be a good proxy.
# We can give the model a rolling sense of this stability via a simple ratio.
df["Proxy_ratio"]   = (df["PM2.5"] / df["AOD"].replace(0, np.nan)).round(4)
# NOTE: Proxy_ratio uses PM2.5 (the target) — do NOT use this as a feature in ML
# It is included here for diagnostic purposes only, excluded from feature list

log("  Proxy_ratio = PM2.5 / AOD  (diagnostic only — NOT a model feature)")

# ── Assemble Dataset B ────────────────────────────────────────────────────────
cols_B = [
    "Date",
    # Target
    "PM2.5",
    # Raw AOD (kept for comparison)
    "AOD",
    # AOD nonlinear transforms
    "AOD_log", "AOD_sqrt", "AOD_sq",
    # CORRECTED AOD features (core contribution)
    "AOD_BLH_corr", "AOD_RH_corr", "AOD_FULL_corr",
    # Correction building blocks
    "BL_proxy", "f_RH",
    # Raw meteorology
    "Wind Speed", "Temperature", "RH", "Solar Rad", "BP", "Rain",
    # Wet scavenging
    "Wet_scavenge",
    # Physical interaction terms
    "AOD_x_RH", "AOD_x_Temp", "AOD_x_WS", "AOD_x_SolRad",
    "AOD_x_Season", "AOD_x_WindPol",
    # Cyclical time
    "Month", "DayOfYear", "Year",
    "Month_sin", "Month_cos", "DOY_sin", "DOY_cos",
    # Cyclical wind direction
    "WindDir_sin", "WindDir_cos",
    # Encoded categoricals
    "Season_ord", "AOD_Loading_ord", "Rain_Status_ord",
    "Humidity_ord", "Temp_ord", "Wind_Polluted",
    "GeoZone_enc", "Station_enc",
    # Coordinates
    "Latitude", "Longitude",
]

cols_B = [c for c in cols_B if c in df.columns]
df_B = df[cols_B].copy()

log(f"\n  Dataset B columns ({len(cols_B)}): {cols_B}")
log(f"  Dataset B shape   : {df_B.shape}")
log(f"  Any NaN in B      : {df_B.drop(columns=['Date']).isnull().any().any()}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 8] Saving outputs")

df_A.to_csv(OUTPUT_A, index=False)
log(f"  Saved Dataset A → '{OUTPUT_A}'  ({df_A.shape[0]:,} rows × {df_A.shape[1]} cols)")

df_B.to_csv(OUTPUT_B, index=False)
log(f"  Saved Dataset B → '{OUTPUT_B}'  ({df_B.shape[0]:,} rows × {df_B.shape[1]} cols)")

# Save log
with open(OUTPUT_LOG, "w") as f:
    f.write("\n".join(log_lines))
log(f"  Saved log       → '{OUTPUT_LOG}'")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  QUICK SANITY REPORT
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("  QUICK SANITY REPORT")
log("=" * 70)

from scipy import stats

log("\nSpearman correlation with PM2.5 — key features:")
check_cols = [
    ("AOD (raw)",          "AOD"),
    ("AOD_FULL_corr",      "AOD_FULL_corr"),
    ("AOD_BLH_corr",       "AOD_BLH_corr"),
    ("AOD_RH_corr",        "AOD_RH_corr"),
    ("BL_proxy",           "BL_proxy"),
    ("f_RH",               "f_RH"),
    ("Temperature",        "Temperature"),
    ("RH",                 "RH"),
    ("Season_ord",         "Season_ord"),
    ("Wind_Polluted",      "Wind_Polluted"),
]

for label, col in check_cols:
    if col in df.columns:
        rho, p = stats.spearmanr(df[col].dropna(), df.loc[df[col].notna(), "PM2.5"])
        sig = "**" if p < 0.001 else ("*" if p < 0.05 else "  ")
        log(f"  {label:22s}: rho={rho:+.4f}  p={p:.4g} {sig}")

log("""
KEY: If AOD_FULL_corr has a higher |rho| with PM2.5 than raw AOD,
     the physical corrections are working.
     Feed both datasets to the ML training script to prove this
     quantitatively with R² comparisons.
""")

log("\n" + "=" * 70)
log("  Feature engineering complete.")
log(f"  → Train ML on '{OUTPUT_A}'  (baseline, naive model)")
log(f"  → Train ML on '{OUTPUT_B}'  (corrected model)")
log("  → Compare R², RMSE to prove correction value")
log("=" * 70)