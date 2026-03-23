import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE  = 'Master_Dataset_Final.csv'
OUTPUT_FILE = 'Master_Dataset_ML_Ready.csv'
REPORT_FILE = 'ML_Readiness_Report.txt'

TARGET = 'PM2.5'

MODEL_FEATURES = [
    'AOD',                    # raw baseline — kept for SHAP comparison
    'AOD_dry',                # f(RH) correction, fixed γ=0.5
    'AOD_corrected',          # season-adaptive two-factor correction — primary
    'f_RH',                   # season-adaptive hygroscopic factor
    'Temp_norm',              # PBLH proxy (Temperature dropped — collinear)
    'RH_squared',             # nonlinear humidity
    'Wind Speed',             # dispersion magnitude
    'Wind_U',                 # eastward wind component
    'Wind_V',                 # northward wind component
    'AODcorr_Wind_interact',  # corrected AOD × wind
    'DOY_sin',                # seasonal cycle sine
    'DOY_cos',                # seasonal cycle cosine
    'Is_Dust_Season',         # pre-monsoon dust flag (Zaman 2021, AAQR 2022)
    'PM2.5_lag1',             # pollution memory
    'AOD_lag1',               # AOD memory
    'AOD_corrected_lag1',     # corrected AOD memory
    'RH_rolling3',            # background humidity state
    'Station_ID',             # station identity (1-4)
    'Season_ID',              # season identity (1-4)
]

META_COLS = ['Date', 'Monitoring_Station', 'Season', 'Split',
             'Year', 'Latitude', 'Longitude']

STATION_ORDER  = ['Agrabad', 'Darus Salam', 'Red Crescent Office', 'Uttar Bagura Road']
SEASON_ORDER   = ['Winter', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon']

report_lines = []
issues_found = []

def rpt(line=''):
    print(line)
    report_lines.append(line)

def flag(severity, message):
    issues_found.append({'Severity': severity, 'Issue': message})
    marker = '[ERROR]' if severity == 'ERROR' else '[WARN] '
    rpt(f"  {marker} {message}")


# ==========================================
# 1. LOAD
# ==========================================
rpt("=" * 70)
rpt("ML READINESS VERIFICATION — FINAL DATASET")
rpt("=" * 70)

df = pd.read_csv(INPUT_FILE, parse_dates=['Date'])
rpt(f"\n  File          : {INPUT_FILE}")
rpt(f"  Rows          : {len(df)}")
rpt(f"  Columns       : {len(df.columns)}")
rpt(f"  Stations      : {sorted(df['Monitoring_Station'].unique())}")
rpt(f"  Date range    : {df['Date'].min().date()} to {df['Date'].max().date()}")
rpt(f"  Train rows    : {(df['Split']=='train').sum()}")
rpt(f"  Test rows     : {(df['Split']=='test').sum()}")


# ==========================================
# CHECK 1: All model features exist
# ==========================================
rpt("\n" + "=" * 70)
rpt("CHECK 1: Feature existence")
rpt("=" * 70)

missing = [f for f in MODEL_FEATURES if f not in df.columns]
present = [f for f in MODEL_FEATURES if f in df.columns]
rpt(f"\n  Features required : {len(MODEL_FEATURES)}")
rpt(f"  Features present  : {len(present)}")
if missing:
    for f in missing:
        flag('ERROR', f"Missing feature column: {f}")
else:
    rpt("  All features present: OK")


# ==========================================
# CHECK 2: NaN audit per feature
# ==========================================
rpt("\n" + "=" * 70)
rpt("CHECK 2: NaN audit")
rpt("=" * 70)

rpt(f"\n  {'Feature':<25} {'NaN':>6}  {'NaN%':>7}  {'Status'}")
rpt(f"  {'-'*52}")
has_nan = False
for col in MODEL_FEATURES + [TARGET]:
    if col not in df.columns:
        continue
    n   = df[col].isna().sum()
    pct = round(100 * n / len(df), 2)
    st  = 'OK' if n == 0 else 'WARNING'
    if n > 0:
        has_nan = True
        flag('WARN', f"{col} has {n} NaN ({pct}%)")
    rpt(f"  {col:<25} {n:>6}  {pct:>6.2f}%  [{st}]")

if has_nan:
    before = len(df)
    df = df.dropna(subset=[c for c in MODEL_FEATURES if c in df.columns]).reset_index(drop=True)
    rpt(f"\n  Dropped {before - len(df)} rows with NaN in model features")
    rpt(f"  Rows remaining: {len(df)}")
else:
    rpt("\n  Zero NaN across all features: OK")


# ==========================================
# CHECK 3: Data types
# ==========================================
rpt("\n" + "=" * 70)
rpt("CHECK 3: Data types — all features must be numeric")
rpt("=" * 70)

rpt(f"\n  {'Feature':<25} {'Dtype':>10}  {'Status'}")
rpt(f"  {'-'*45}")
for col in MODEL_FEATURES:
    if col not in df.columns:
        continue
    dtype  = str(df[col].dtype)
    is_num = pd.api.types.is_numeric_dtype(df[col])
    status = 'OK' if is_num else 'NOT NUMERIC'
    if not is_num:
        flag('ERROR', f"{col} is not numeric: {dtype}")
    rpt(f"  {col:<25} {dtype:>10}  [{status}]")


# ==========================================
# CHECK 4: Value ranges — physical sanity
# ==========================================
rpt("\n" + "=" * 70)
rpt("CHECK 4: Value range sanity")
rpt("=" * 70)

EXPECTED_RANGES = {
    'AOD'                  : (0,    5.0,   'MODIS valid range'),
    'AOD_dry'              : (0,    5.0,   'hygroscopic corrected'),
    'AOD_corrected'        : (0,    10.0,  'season-adaptive two-factor'),
    'f_RH'                 : (1.0,  10.0,  'growth factor >= 1'),
    'Temp_norm'            : (0.1,  5.0,   'normalised temperature'),
    'RH_squared'           : (0,    10000, 'RH% squared'),
    'Wind Speed'           : (0,    50,    'physical limit'),
    'DOY_sin'              : (-1,   1,     'cyclic encoding'),
    'DOY_cos'              : (-1,   1,     'cyclic encoding'),
    'Is_Dust_Season'       : (0,    1,     'binary dust flag'),
    'Station_ID'           : (1,    4,     'must be 1-4'),
    'Season_ID'            : (1,    4,     'must be 1-4'),
    'PM2.5'                : (0,    999,   'ground measurement'),
}

rpt(f"\n  {'Feature':<25} {'Min':>10}  {'Max':>10}  {'Expected range':<20}  Status")
rpt(f"  {'-'*80}")
for col, (lo, hi, note) in EXPECTED_RANGES.items():
    if col not in df.columns:
        continue
    mn = df[col].min()
    mx = df[col].max()
    ok = (mn >= lo) and (mx <= hi)
    st = 'OK' if ok else 'OUT OF RANGE'
    if not ok:
        flag('WARN', f"{col} outside expected [{lo}, {hi}]: actual [{mn:.3f}, {mx:.3f}]")
    rpt(f"  {col:<25} {mn:>10.4f}  {mx:>10.4f}  [{lo}, {hi}] {note:<15}  [{st}]")


# ==========================================
# CHECK 5: Reality cross-checks
#   Known physical truths for Bangladesh
# ==========================================
rpt("\n" + "=" * 70)
rpt("CHECK 5: Physical reality cross-checks")
rpt("=" * 70)

rpt(f"\n  [A] PM2.5 by station — Dhaka (Darus Salam) must rank highest")
rpt(f"  {'Station':<25} {'Mean':>8}  {'Median':>8}  {'Std':>8}  {'Max':>8}  Rank")
rpt(f"  {'-'*68}")
pm_rank = df.groupby('Monitoring_Station')['PM2.5'].mean().sort_values(ascending=False)
for rank, (stn, val) in enumerate(pm_rank.items(), 1):
    grp = df[df['Monitoring_Station'] == stn]['PM2.5']
    rpt(f"  {stn:<25} {grp.mean():>8.1f}  {grp.median():>8.1f}  "
        f"{grp.std():>8.1f}  {grp.max():>8.1f}  #{rank}")
if pm_rank.index[0] != 'Darus Salam':
    flag('WARN', f"Darus Salam not highest PM2.5 — check data")
else:
    rpt("  Dhaka (Darus Salam) ranks highest: OK")

rpt(f"\n  [B] Seasonal pattern — Monsoon highest RH, Winter highest PM2.5")
rpt(f"  {'Season':<15} {'Mean RH':>9}  {'Mean PM2.5':>11}  {'Mean Temp_norm':>14}  "
    f"{'Mean AOD':>10}  {'Mean AOD_corr':>14}")
rpt(f"  {'-'*78}")
for ssn in SEASON_ORDER:
    g = df[df['Season'] == ssn]
    rpt(f"  {ssn:<15} {g['RH'].mean():>9.1f}  {g['PM2.5'].mean():>11.1f}  "
        f"{g['Temp_norm'].mean():>14.3f}  {g['AOD'].mean():>10.4f}  "
        f"{g['AOD_corrected'].mean():>14.4f}")

monsoon_rh  = df[df['Season']=='Monsoon']['RH'].mean()
winter_rh   = df[df['Season']=='Winter']['RH'].mean()
winter_pm   = df[df['Season']=='Winter']['PM2.5'].mean()
monsoon_pm  = df[df['Season']=='Monsoon']['PM2.5'].mean()

if monsoon_rh < winter_rh:
    flag('WARN', "Monsoon RH not higher than Winter — check seasonal labelling")
else:
    rpt("  Monsoon RH > Winter RH: OK")

if winter_pm < monsoon_pm:
    flag('WARN', "Winter PM2.5 not higher than Monsoon — check data")
else:
    rpt("  Winter PM2.5 > Monsoon PM2.5: OK")

rpt(f"\n  [C] f(RH) should be highest in Monsoon (season-adaptive γ)")
rpt(f"  {'Season':<15} {'γ':>5}  {'mean f(RH)':>11}  {'mean Temp_norm':>15}  "
    f"{'mean divisor':>13}  Note")
rpt(f"  {'-'*66}")
for ssn in SEASON_ORDER:
    g       = df[df['Season'] == ssn]
GAMMA_BY_SEASON = {'Winter':0.50,'Pre-Monsoon':0.10,'Monsoon':0.40,'Post-Monsoon':0.45}
for ssn in SEASON_ORDER:
    g        = df[df['Season'] == ssn]
    div_mean = (g['f_RH'] * g['Temp_norm']).mean()
    gval     = GAMMA_BY_SEASON.get(ssn, 0.5)
    note     = '<-- largest correction' if div_mean == max(
        (df[df['Season']==s]['f_RH'] * df[df['Season']==s]['Temp_norm']).mean()
        for s in SEASON_ORDER) else ''
    rpt(f"  {ssn:<15} {gval:>5.2f}  {g['f_RH'].mean():>11.3f}  "
        f"{g['Temp_norm'].mean():>15.3f}  {div_mean:>13.3f}  {note}")

# Pre-monsoon should have lowest f(RH) due to γ=0.10 (dust aerosols)
pre_frh  = df[df['Season']=='Pre-Monsoon']['f_RH'].mean()
win_frh  = df[df['Season']=='Winter']['f_RH'].mean()
if pre_frh < win_frh:
    rpt(f"\n  Pre-monsoon f(RH) ({pre_frh:.3f}) < Winter f(RH) ({win_frh:.3f}): OK")
    rpt(f"  Season-adaptive γ working correctly — dust season correction suppressed")
else:
    flag('WARN', f"Pre-monsoon f(RH) not suppressed — check γ=0.10 applied correctly")

rpt(f"\n  [D] AOD_corrected < AOD_raw when divisor > 1")
rpt(f"  [E] Is_Dust_Season flag — must be 1 only for Pre-Monsoon rows")

rpt(f"\n  [D] AOD_corrected < AOD_raw when divisor > 1 (correction reduces inflated AOD)")
for stn in STATION_ORDER:
    g = df[df['Monitoring_Station'] == stn]
    pct_reduced = (g['AOD_corrected'] < g['AOD']).mean() * 100
    rpt(f"  {stn:<25} {pct_reduced:.1f}% of days: AOD_corrected < AOD_raw")
    if pct_reduced < 50:
        flag('WARN', f"{stn}: less than 50% of days show downward correction")

rpt(f"\n  [E] Is_Dust_Season flag integrity")
for ssn in SEASON_ORDER:
    g    = df[df['Season'] == ssn]
    flag_mean = g['Is_Dust_Season'].mean()
    expected  = 1.0 if ssn == 'Pre-Monsoon' else 0.0
    ok        = abs(flag_mean - expected) < 0.01
    status    = 'OK' if ok else 'ERROR'
    if not ok:
        flag('ERROR', f"Is_Dust_Season incorrect for {ssn}: mean={flag_mean:.3f}, expected={expected}")
    rpt(f"  {ssn:<15} Is_Dust_Season mean={flag_mean:.1f}  expected={expected:.1f}  [{status}]")


# ==========================================
# CHECK 6: Core result — does correction
#   improve correlation with PM2.5?
#   This is the heart of the paper.
# ==========================================
rpt("\n" + "=" * 70)
rpt("CHECK 6: Correction effectiveness — the core result")
rpt("=" * 70)

rpt(f"\n  Overall correlations with PM2.5:")
rpt(f"  {'Feature':<25} {'r (all data)':>14}  Note")
rpt(f"  {'-'*52}")
aod_cols = ['AOD', 'AOD_dry', 'AOD_corrected']
for col in aod_cols:
    r = df[[col, TARGET]].dropna().corr().iloc[0, 1]
    rpt(f"  {col:<25} {r:>14.4f}")

r_raw  = df[['AOD',           TARGET]].dropna().corr().iloc[0, 1]
r_dry  = df[['AOD_dry',       TARGET]].dropna().corr().iloc[0, 1]
r_corr = df[['AOD_corrected', TARGET]].dropna().corr().iloc[0, 1]

rpt(f"\n  Improvement from AOD_raw to AOD_dry      : {r_dry  - r_raw:+.4f}  (f(RH) alone)")
rpt(f"  Improvement from AOD_raw to AOD_corrected: {r_corr - r_raw:+.4f}  (two-factor)")
rpt(f"  Improvement from AOD_dry to AOD_corrected: {r_corr - r_dry:+.4f}  (Temp_norm contribution)")

if r_corr > r_dry > r_raw:
    rpt("\n  Correction hierarchy confirmed: AOD_corr > AOD_dry > AOD_raw")
    rpt("  Both correction factors contribute independently: OK")
elif r_corr > r_raw:
    rpt("\n  Two-factor correction improves on raw AOD: OK")
    rpt("  Note: AOD_dry may not improve on raw — humidity correction alone insufficient")
else:
    flag('WARN', "AOD_corrected does not improve correlation over AOD_raw overall")

rpt(f"\n  Per-station × per-season breakdown:")
rpt(f"  {'Station':<25} {'Season':<15} {'r(AOD)':>8}  "
    f"{'r(AOD_dry)':>11}  {'r(AOD_corr)':>12}  {'Best':>12}")
rpt(f"  {'-'*88}")
improved_count = 0
total_count    = 0
for stn in STATION_ORDER:
    for ssn in SEASON_ORDER:
        g = df[(df['Monitoring_Station']==stn) & (df['Season']==ssn)]
        if len(g) < 10:
            continue
        r1 = g[['AOD',           TARGET]].corr().iloc[0, 1]
        r2 = g[['AOD_dry',       TARGET]].corr().iloc[0, 1]
        r3 = g[['AOD_corrected', TARGET]].corr().iloc[0, 1]
        best = max([('AOD_raw', r1), ('AOD_dry', r2), ('AOD_corrected', r3)],
                   key=lambda x: abs(x[1]))
        if best[0] == 'AOD_corrected':
            improved_count += 1
        total_count += 1
        marker = ' <--' if best[0] == 'AOD_corrected' else ''
        rpt(f"  {stn:<25} {ssn:<15} {r1:>8.4f}  {r2:>11.4f}  "
            f"{r3:>12.4f}  {best[0]}{marker}")

rpt(f"\n  AOD_corrected is best in {improved_count}/{total_count} station-season groups "
    f"({round(100*improved_count/max(total_count,1))}%)")


# ==========================================
# CHECK 7: Multicollinearity screen
# ==========================================
rpt("\n" + "=" * 70)
rpt("CHECK 7: Multicollinearity screen (|r| > 0.92 flagged)")
rpt("=" * 70)

feat_df  = df[[c for c in MODEL_FEATURES if c in df.columns]].dropna()
corr_mat = feat_df.corr().abs()
flagged  = []
cols     = corr_mat.columns.tolist()
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        r = corr_mat.iloc[i, j]
        if r > 0.92:
            flagged.append((cols[i], cols[j], round(r, 4)))

if flagged:
    rpt(f"\n  {len(flagged)} highly correlated pairs (|r| > 0.92):")
    rpt(f"  {'Feature A':<25} {'Feature B':<25} {'|r|':>6}  Action")
    rpt(f"  {'-'*72}")
    for a, b, r in sorted(flagged, key=lambda x: -x[2]):
        action = 'Keep both — interpretable components' if r < 0.97 else 'Consider dropping one'
        rpt(f"  {a:<25} {b:<25} {r:>6.4f}  {action}")
        if r > 0.97:
            flag('WARN', f"High collinearity: {a} vs {b} (|r|={r})")
else:
    rpt("\n  No extreme collinearity detected: OK")


# ==========================================
# CHECK 8: Train/test integrity
# ==========================================
rpt("\n" + "=" * 70)
rpt("CHECK 8: Train/test split integrity")
rpt("=" * 70)

rpt(f"\n  {'Station':<25} {'Season':<15} {'Train':>6}  {'Test':>6}  {'Test%':>7}")
rpt(f"  {'-'*62}")
for (stn, ssn), grp in df.groupby(['Monitoring_Station', 'Season']):
    tr  = (grp['Split'] == 'train').sum()
    te  = (grp['Split'] == 'test').sum()
    pct = round(100 * te / len(grp), 1)
    if te == 0:
        flag('ERROR', f"No test rows for {stn} / {ssn}")
    rpt(f"  {stn:<25} {ssn:<15} {tr:>6}  {te:>6}  {pct:>6.1f}%")

test_stns = df[df['Split']=='test']['Monitoring_Station'].unique()
test_ssns = df[df['Split']=='test']['Season'].unique()
rpt(f"\n  Stations in test : {sorted(test_stns)}")
rpt(f"  Seasons in test  : {sorted(test_ssns)}")

if len(test_stns) < 4:
    flag('ERROR', f"Only {len(test_stns)}/4 stations represented in test set")
if len(test_ssns) < 4:
    flag('ERROR', f"Only {len(test_ssns)}/4 seasons represented in test set")


# ==========================================
# CHECK 9: Correlation with target
# ==========================================
rpt("\n" + "=" * 70)
rpt("CHECK 9: Feature correlations with PM2.5")
rpt("=" * 70)

corr = (
    df[[c for c in MODEL_FEATURES if c in df.columns] + [TARGET]]
    .corr()[TARGET]
    .drop(TARGET)
    .sort_values(key=abs, ascending=False)
)
rpt(f"\n  {'Feature':<25} {'r':>8}  {'|r|':>8}  Strength")
rpt(f"  {'-'*55}")
for feat, r in corr.items():
    strength = ('Strong'   if abs(r) > 0.5 else
                'Moderate' if abs(r) > 0.3 else
                'Weak'     if abs(r) > 0.1 else 'Negligible')
    rpt(f"  {feat:<25} {r:>8.4f}  {abs(r):>8.4f}  {strength}")


# ==========================================
# CHECK 10: Distribution check
# ==========================================
rpt("\n" + "=" * 70)
rpt("CHECK 10: Distribution check (skewness)")
rpt("=" * 70)

rpt(f"\n  {'Feature':<25} {'Skew':>8}  {'Kurt':>8}  Note")
rpt(f"  {'-'*60}")
for col in ['PM2.5', 'AOD', 'AOD_dry', 'AOD_corrected', 'RH_squared',
            'Temp_norm', 'Wind Speed']:
    if col not in df.columns:
        continue
    sk = df[col].skew()
    ku = df[col].kurtosis()
    note = ('HIGH SKEW — log transform may help' if abs(sk) > 2 else
            'Moderate skew — acceptable for RF'  if abs(sk) > 1 else
            'Near-normal')
    rpt(f"  {col:<25} {sk:>8.3f}  {ku:>8.3f}  {note}")


# ==========================================
# ISSUE SUMMARY
# ==========================================
rpt("\n" + "=" * 70)
rpt("ISSUE SUMMARY")
rpt("=" * 70)

errors   = [i for i in issues_found if i['Severity'] == 'ERROR']
warnings = [i for i in issues_found if i['Severity'] == 'WARN']

rpt(f"\n  Errors   : {len(errors)}")
rpt(f"  Warnings : {len(warnings)}")

if errors:
    rpt("\n  ERRORS (must fix before modeling):")
    for i in errors:
        rpt(f"    [ERROR] {i['Issue']}")

if warnings:
    rpt("\n  Warnings (review before modeling):")
    for i in warnings:
        rpt(f"    [WARN]  {i['Issue']}")

if not errors:
    rpt("\n  No blocking errors found.")


# ==========================================
# SAVE
# ==========================================
final_cols = META_COLS + [TARGET] + [c for c in MODEL_FEATURES if c in df.columns]
df_final   = df[[c for c in final_cols if c in df.columns]]
df_final.to_csv(OUTPUT_FILE, index=False)

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

rpt("\n" + "=" * 70)
if not errors:
    rpt("DATASET IS ML-READY")
else:
    rpt(f"DATASET HAS {len(errors)} BLOCKING ERROR(S) — FIX BEFORE MODELING")
rpt("=" * 70)
rpt(f"\n  Saved: '{OUTPUT_FILE}'")
rpt(f"  Rows : {len(df_final)}")
rpt(f"  Report: '{REPORT_FILE}'")
rpt(f"\n  MODEL_FEATURES = {MODEL_FEATURES}")
rpt(f"  TARGET         = '{TARGET}'")