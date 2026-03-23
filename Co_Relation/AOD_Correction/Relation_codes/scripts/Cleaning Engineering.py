import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE  = 'Master_Dataset_Daily_Raw.csv'
OUTPUT_FILE = 'Master_Dataset_Final.csv'

SELECTED_STATIONS = [
    'Agrabad', 'Darus Salam', 'Red Crescent Office', 'Uttar Bagura Road'
]

STATION_ID_MAP = {
    'Agrabad': 1, 'Darus Salam': 2,
    'Red Crescent Office': 3, 'Uttar Bagura Road': 4,
}
SEASON_ID_MAP = {
    'Winter': 1, 'Pre-Monsoon': 2, 'Monsoon': 3, 'Post-Monsoon': 4,
}

PHYSICAL_BOUNDS = {
    'PM2.5'      : (0,   999),
    'AOD'        : (0,   5.0),
    'RH'         : (0,   100),
    'Temperature': (-5,  45),
    'Wind Speed' : (0,   50),
    'Wind Dir'   : (0,   360),
    'Solar Rad'  : (0,   1500),
    'BP'         : (900, 1100),
    'Rain'       : (0,   500),
}

IQR_MULTIPLIER = 3.0
GAP_SHORT_MAX  = 3
GAP_MEDIUM_MAX = 15
TRAIN_YEARS    = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
TEST_FRACTION  = 0.20

# ==========================================
# SEASON-ADAPTIVE HYGROSCOPIC GROWTH FACTOR
# Literature basis:
#   Winter (fine-mode hygroscopic)    γ=0.50 — Levy et al. 2007
#   Pre-monsoon (dust dominated)      γ=0.10 — dust non-hygroscopic
#                                              Zaman et al. 2021, AAQR 2022
#   Monsoon (mixed + marine)          γ=0.40 — hygroscopic but diluted
#                                              by rainfall washout
#   Post-monsoon (transitional        γ=0.45 — mostly fine-mode
#                 fine-mode returns)           returning after monsoon
# ==========================================
GAMMA_BY_SEASON = {
    'Winter'      : 0.50,
    'Pre-Monsoon' : 0.10,
    'Monsoon'     : 0.40,
    'Post-Monsoon': 0.45,
}

IMPUTE_COLS  = ['PM2.5', 'AOD', 'RH', 'Temperature', 'Wind Speed', 'Wind Dir']
REQUIRED_MET = ['PM2.5', 'Wind Speed', 'Wind Dir', 'Temperature', 'RH']
OPTIONAL_MET = ['Solar Rad', 'BP', 'Rain', 'V Wind Speed']


# ==========================================
# HELPERS
# ==========================================
def classify_and_impute(series, season_series):
    s      = series.copy()
    is_nan = s.isna()
    result = pd.Series('ok', index=s.index)
    in_gap, gap_start = False, None
    positions = list(s.index)

    for i, idx in enumerate(positions):
        if is_nan[idx] and not in_gap:
            in_gap, gap_start = True, i
        elif not is_nan[idx] and in_gap:
            gap_len = i - gap_start
            label   = ('short'  if gap_len <= GAP_SHORT_MAX  else
                       'medium' if gap_len <= GAP_MEDIUM_MAX else 'long')
            for j in range(gap_start, i):
                result.iloc[j] = label
            in_gap = False
    if in_gap:
        gap_len = len(positions) - gap_start
        label   = ('short'  if gap_len <= GAP_SHORT_MAX  else
                   'medium' if gap_len <= GAP_MEDIUM_MAX else 'long')
        for j in range(gap_start, len(positions)):
            result.iloc[j] = label

    short_idx  = result[result == 'short'].index
    medium_idx = result[result == 'medium'].index
    long_idx   = result[result == 'long'].index

    if len(short_idx) > 0:
        filled = s.fillna(s.rolling(window=7, min_periods=1, center=True).median())
        s.update(filled[short_idx])
    if len(medium_idx) > 0:
        tmp = pd.DataFrame({'val': s, 'season': season_series})
        seas_mean = tmp.groupby('season')['val'].transform('mean')
        s.update(seas_mean[medium_idx])

    return s, len(short_idx), len(medium_idx), len(long_idx)


# ==========================================
# 1. LOAD & FILTER
# ==========================================
print("=" * 65)
print("STEP 1: Load and filter")
print("=" * 65)
df = pd.read_csv(INPUT_FILE, parse_dates=['Date'])
df = df[df['Monitoring_Station'].isin(SELECTED_STATIONS)].copy()
df = df.sort_values(['Monitoring_Station', 'Date']).reset_index(drop=True)
df['Year']   = df['Date'].dt.year
df['Month']  = df['Date'].dt.month
print(f"  Rows: {len(df)} | Stations: {list(df['Monitoring_Station'].unique())}")


# ==========================================
# 2. PHYSICAL BOUNDS
# ==========================================
print("\n" + "=" * 65)
print("STEP 2: Physical bounds cleaning")
print("=" * 65)
for col, (lo, hi) in PHYSICAL_BOUNDS.items():
    if col not in df.columns:
        continue
    mask = (df[col] < lo) | (df[col] > hi)
    n    = mask.sum()
    if n > 0:
        df.loc[mask, col] = np.nan
        print(f"  {col:<20} {n:>4} values flagged → NaN")
    else:
        print(f"  {col:<20} OK")


# ==========================================
# 3. IQR OUTLIER DETECTION
# ==========================================
print("\n" + "=" * 65)
print("STEP 3: IQR outlier detection (3.0× per station × season)")
print("=" * 65)
for col in ['PM2.5', 'AOD', 'RH', 'Wind Speed', 'Temperature']:
    if col not in df.columns:
        continue
    total = 0
    for (stn, ssn), grp in df.groupby(['Monitoring_Station', 'Season']):
        q1  = grp[col].quantile(0.25)
        q3  = grp[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - IQR_MULTIPLIER * iqr
        hi  = q3 + IQR_MULTIPLIER * iqr
        mask = (df.loc[grp.index, col] < lo) | (df.loc[grp.index, col] > hi)
        n    = mask.sum()
        if n > 0:
            df.loc[grp.index[mask], col] = np.nan
            total += n
    print(f"  {col:<20} {total:>4} outliers flagged")


# ==========================================
# 4. GAP IMPUTATION (all variables)
# ==========================================
print("\n" + "=" * 65)
print("STEP 4: Gap imputation")
print("=" * 65)
for station in SELECTED_STATIONS:
    mask   = df['Monitoring_Station'] == station
    stn_df = df[mask].copy().sort_values('Date')
    print(f"\n  [{station}]")
    for col in IMPUTE_COLS:
        if col not in stn_df.columns:
            continue
        n_before = stn_df[col].isna().sum()
        if n_before == 0:
            continue
        stn_df[col], ns, nm, nl = classify_and_impute(stn_df[col], stn_df['Season'])
        n_after = stn_df[col].isna().sum()
        print(f"    {col:<20} before:{n_before:>4}  "
              f"short:{ns:>3}  medium:{nm:>3}  long(unfilled):{nl:>3}  after:{n_after:>4}")
    df.loc[mask, IMPUTE_COLS] = stn_df[IMPUTE_COLS].values

rows_before = len(df)
df = df.dropna(subset=['PM2.5', 'AOD']).reset_index(drop=True)
print(f"\n  Dropped {rows_before - len(df)} rows (long PM2.5/AOD gaps)")


# ==========================================
# 5. SEASON-ADAPTIVE TWO-FACTOR AOD
#    CORRECTION
#
#  Formula:
#    AOD_corrected = AOD / (f_seasonal(RH) × Temp_norm)
#
#  Factor 1 — f_seasonal(RH):
#    Season-adaptive hygroscopic growth
#    γ varies by aerosol type per season
#    Literature: Levy 2007, Zaman 2021,
#                AAQR 2022, Water Air Soil
#                Pollution 2025
#
#  Factor 2 — Temp_norm:
#    Temperature-based PBLH proxy
#    Normalised to training-period mean
#    per station
#    Literature: Frontiers 2022,
#                India MODIS 2019
#
#  γ by season:
#    Winter (0.50)      — fine-mode hygroscopic
#    Pre-monsoon (0.10) — dust dominated,
#                         near non-hygroscopic
#    Monsoon (0.40)     — mixed hygroscopic
#    Post-monsoon (0.45)— transitional fine-mode
# ==========================================
print("\n" + "=" * 65)
print("STEP 5: Season-adaptive two-factor AOD correction")
print("=" * 65)

print("\n  Season-adaptive γ values (hygroscopic growth exponent):")
for ssn, g in GAMMA_BY_SEASON.items():
    print(f"    {ssn:<15} γ = {g}  "
          f"{'[dust — near non-hygroscopic]' if g < 0.2 else '[hygroscopic]'}")

# Factor 1: season-adaptive f(RH)
rh_safe = df['RH'].clip(upper=99.5)
df['gamma'] = df['Season'].map(GAMMA_BY_SEASON)
df['f_RH']  = (1 - rh_safe / 100) ** (-df['gamma'])
df['f_RH']  = df['f_RH'].clip(upper=10)

# AOD_dry uses fixed γ=0.5 (hygroscopic baseline — for SHAP comparison)
df['AOD_dry'] = (df['AOD'] / ((1 - rh_safe / 100) ** (-0.5))).clip(
    lower=df['AOD'] * 0.05, upper=df['AOD'] * 3.0
)

print(f"\n  f(RH) by season (season-adaptive γ):")
print(f"  {'Season':<15} {'γ':>5}  {'mean RH':>9}  {'mean f(RH)':>11}  "
      f"{'AOD inflation':>14}  Physical interpretation")
print(f"  {'-'*78}")
for ssn in ['Winter', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon']:
    g    = df[df['Season'] == ssn]
    frh  = g['f_RH'].mean()
    gval = GAMMA_BY_SEASON[ssn]
    note = ('Dust — minimal hygroscopic correction'  if gval < 0.2 else
            'Mixed — moderate correction'            if gval < 0.45 else
            'Fine-mode — full hygroscopic correction')
    print(f"  {ssn:<15} {gval:>5.2f}  {g['RH'].mean():>9.1f}%  {frh:>11.3f}  "
          f"{(frh-1)*100:>13.0f}%  {note}")

# Factor 2: temperature-based PBLH proxy
# Normalised to training-period mean per station (no data leakage)
train_temp_mean = (
    df[df['Year'].isin(TRAIN_YEARS)]
    .groupby('Monitoring_Station')['Temperature']
    .mean()
    .rename('Temp_train_mean')
)
df = df.merge(train_temp_mean, on='Monitoring_Station', how='left')
df['Temp_norm'] = (df['Temperature'] / df['Temp_train_mean']).clip(0.1, 5.0)

# Apply two-factor correction
df['AOD_corrected'] = df['AOD'] / (df['f_RH'] * df['Temp_norm'])
df['AOD_corrected'] = df['AOD_corrected'].clip(
    lower=df['AOD'] * 0.05, upper=df['AOD'] * 3.0
)

print(f"\n  Correction effectiveness — season-adaptive vs fixed γ:")
print(f"  {'Season':<15} {'r(AOD_raw)':>11}  {'r(AOD_dry)':>11}  "
      f"{'r(AOD_corr)':>12}  {'Best':>14}  Delta vs raw")
print(f"  {'-'*82}")
for ssn in ['Winter', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon']:
    g  = df[df['Season'] == ssn]
    r1 = g[['AOD',           'PM2.5']].dropna().corr().iloc[0, 1]
    r2 = g[['AOD_dry',       'PM2.5']].dropna().corr().iloc[0, 1]
    r3 = g[['AOD_corrected', 'PM2.5']].dropna().corr().iloc[0, 1]
    best = max([('AOD_raw', r1), ('AOD_dry', r2), ('AOD_corr', r3)],
               key=lambda x: abs(x[1]))
    print(f"  {ssn:<15} {r1:>11.4f}  {r2:>11.4f}  {r3:>12.4f}  "
          f"{best[0]:>14}  {r3-r1:>+.4f}")

r_raw  = df[['AOD',           'PM2.5']].dropna().corr().iloc[0, 1]
r_dry  = df[['AOD_dry',       'PM2.5']].dropna().corr().iloc[0, 1]
r_corr = df[['AOD_corrected', 'PM2.5']].dropna().corr().iloc[0, 1]
print(f"\n  Overall: r_raw={r_raw:.4f}  r_dry={r_dry:.4f}  "
      f"r_corrected={r_corr:.4f}  delta={r_corr-r_raw:+.4f}")


# ==========================================
# 6. FEATURE ENGINEERING
# ==========================================
print("\n" + "=" * 65)
print("STEP 6: Feature engineering")
print("=" * 65)

# Temporal cyclic
df['DOY']     = df['Date'].dt.dayofyear
df['DOY_sin'] = np.sin(2 * np.pi * df['DOY'] / 365)
df['DOY_cos'] = np.cos(2 * np.pi * df['DOY'] / 365)

# Wind decomposition
wind_dir_rad = np.deg2rad(df['Wind Dir'])
df['Wind_U'] = df['Wind Speed'] * np.cos(wind_dir_rad)
df['Wind_V'] = df['Wind Speed'] * np.sin(wind_dir_rad)

# Nonlinear humidity
df['RH_squared'] = df['RH'] ** 2

# Interaction features
df['AODcorr_Wind_interact'] = df['AOD_corrected'] * df['Wind Speed']
df['AODcorr_RH_residual']   = df['AOD_corrected'] * df['RH']

# Dust season flag — literature-grounded binary feature
# Pre-monsoon = dust dominated (Zaman et al. 2021, AAQR 2022)
df['Is_Dust_Season'] = (df['Season'] == 'Pre-Monsoon').astype(int)

# Categorical encodings
df['Station_ID'] = df['Monitoring_Station'].map(STATION_ID_MAP)
df['Season_ID']  = df['Season'].map(SEASON_ID_MAP)

# Lag features — per station, date sorted
print("  Computing lag features...")
lag_frames = []
for station in SELECTED_STATIONS:
    stn_df = df[df['Monitoring_Station'] == station].copy().sort_values('Date')
    stn_df['PM2.5_lag1']         = stn_df['PM2.5'].shift(1)
    stn_df['AOD_lag1']           = stn_df['AOD'].shift(1)
    stn_df['AOD_corrected_lag1'] = stn_df['AOD_corrected'].shift(1)
    stn_df['RH_rolling3']        = stn_df['RH'].rolling(window=3, min_periods=1).mean()
    lag_frames.append(stn_df)

df = pd.concat(lag_frames, ignore_index=True)
df = df.sort_values(['Monitoring_Station', 'Date']).reset_index(drop=True)

rows_before = len(df)
df = df.dropna(subset=['PM2.5_lag1', 'AOD_lag1']).reset_index(drop=True)
print(f"  Dropped {rows_before - len(df)} rows (lag NaN, first row per station)")

print(f"\n  New features created:")
new_feats = ['AOD_dry', 'AOD_corrected', 'f_RH', 'Temp_norm', 'DOY_sin',
             'DOY_cos', 'Wind_U', 'Wind_V', 'RH_squared', 'AODcorr_Wind_interact',
             'AODcorr_RH_residual', 'Is_Dust_Season', 'PM2.5_lag1',
             'AOD_lag1', 'AOD_corrected_lag1', 'RH_rolling3', 'Station_ID', 'Season_ID']
for f in new_feats:
    print(f"    {f}")


# ==========================================
# 7. TRAIN / TEST SPLIT
#    Last 20% of each station × season
#    group → test (temporal integrity)
# ==========================================
print("\n" + "=" * 65)
print("STEP 7: Train/test split")
print("=" * 65)

df['Split'] = 'train'
for (stn, ssn), grp in df.groupby(['Monitoring_Station', 'Season']):
    grp_sorted = grp.sort_values('Date')
    n_test     = max(1, int(np.floor(len(grp_sorted) * TEST_FRACTION)))
    df.loc[grp_sorted.index[-n_test:], 'Split'] = 'test'

print(f"\n  {'Station':<25} {'Season':<15} {'Train':>6}  {'Test':>6}  {'Test%':>7}")
print(f"  {'-'*62}")
for (stn, ssn), grp in df.groupby(['Monitoring_Station', 'Season']):
    tr  = (grp['Split'] == 'train').sum()
    te  = (grp['Split'] == 'test').sum()
    pct = round(100 * te / len(grp), 1)
    print(f"  {stn:<25} {ssn:<15} {tr:>6}  {te:>6}  {pct:>6.1f}%")

print(f"\n  Total train: {(df['Split']=='train').sum()}")
print(f"  Total test : {(df['Split']=='test').sum()}")


# ==========================================
# 8. FINAL NaN CHECK & SAVE
# ==========================================
MODEL_FEATURES = [
    'AOD',                    # raw baseline
    'AOD_dry',                # f(RH) correction, fixed γ=0.5
    'AOD_corrected',          # season-adaptive two-factor correction
    'f_RH',                   # season-adaptive hygroscopic factor
    'Temp_norm',              # PBLH proxy
    'RH_squared',             # nonlinear humidity
    'Wind Speed',             # dispersion magnitude
    'Wind_U',                 # eastward wind
    'Wind_V',                 # northward wind
    'AODcorr_Wind_interact',  # corrected AOD × wind
    'DOY_sin',                # seasonal cycle
    'DOY_cos',
    'Is_Dust_Season',         # dust aerosol flag (pre-monsoon)
    'PM2.5_lag1',             # pollution memory
    'AOD_lag1',               # AOD memory
    'AOD_corrected_lag1',     # corrected AOD memory
    'RH_rolling3',            # background humidity
    'Station_ID',             # station identity
    'Season_ID',              # season identity
]

print("\n" + "=" * 65)
print("STEP 8: Final NaN check")
print("=" * 65)
has_nan = False
for col in MODEL_FEATURES + ['PM2.5']:
    if col not in df.columns:
        print(f"  MISSING: {col}")
        continue
    n = df[col].isna().sum()
    if n > 0:
        has_nan = True
        print(f"  WARNING: {col} has {n} NaN")
if has_nan:
    before = len(df)
    df = df.dropna(subset=[c for c in MODEL_FEATURES if c in df.columns])
    df = df.reset_index(drop=True)
    print(f"  Dropped {before - len(df)} rows after final NaN check")
else:
    print("  All model features: ZERO NaN — clean")

# Drop helper and redundant columns
DROP_COLS = ['Wind Dir', 'V Wind Speed', 'Solar Rad', 'BP', 'Rain',
             'DOY', 'Month', 'Temp_train_mean', 'gamma']
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

META_COLS  = ['Date', 'Monitoring_Station', 'Season', 'Split',
              'Year', 'Latitude', 'Longitude']
col_order  = META_COLS + ['PM2.5'] + MODEL_FEATURES
extra_cols = [c for c in df.columns if c not in col_order]
df_final   = df[[c for c in col_order if c in df.columns] + extra_cols]
df_final   = df_final.sort_values(['Monitoring_Station', 'Date']).reset_index(drop=True)
df_final.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("DATASET COMPLETE")
print("=" * 65)
print(f"\n  Output file    : '{OUTPUT_FILE}'")
print(f"  Total rows     : {len(df_final)}")
print(f"  Model features : {len(MODEL_FEATURES)}")
print(f"  Train rows     : {(df_final['Split']=='train').sum()}")
print(f"  Test rows      : {(df_final['Split']=='test').sum()}")
print(f"\n  Correction formula:")
print(f"    AOD_corrected = AOD / (f_seasonal(RH) × Temp_norm)")
print(f"    γ: Winter=0.50, Pre-Monsoon=0.10, Monsoon=0.40, Post-Monsoon=0.45")
print(f"    Literature basis: Levy 2007, Zaman 2021, AAQR 2022, Frontiers 2022")
print(f"\n  MODEL_FEATURES = {MODEL_FEATURES}")
print(f"  TARGET = 'PM2.5'")
print(f"\n  Ready for ML model training.")