import pandas as pd
import numpy as np
import time

start_timer = time.time()

# ==========================================
# 1. FILE PATHS & CONFIGURATION
# ==========================================
aod_file      = 'AOD-14-21-daywise.csv'
doe_file      = 'DoE CAMS Air Qualtiy Data.xlsx'
location_file = 'site_location.xlsx'
output_file   = 'Master_Dataset_Daily_Raw.csv'
audit_file    = 'Station_Audit_Report.csv'   # full three-source audit
report_file   = 'Station_Completeness_Report.csv'  # per-variable detail

start_date = pd.to_datetime('2014-01-01')
end_date   = pd.to_datetime('2021-12-31')

# Variables that MUST have real data — station excluded if ANY of these is empty
REQUIRED_VARS = ['PM2.5', 'Wind Speed', 'Wind Dir', 'Temperature', 'RH']
OPTIONAL_VARS = ['Solar Rad', 'BP', 'Rain', 'V Wind Speed']
ALL_TARGET_COLS = REQUIRED_VARS + OPTIONAL_VARS

# Minimum temporal coverage threshold (days) for final selection
MIN_DAYS_THRESHOLD = 500

# AOD stations — exactly the 10 columns in the AOD CSV
AOD_STATIONS = [
    'Agrabad', 'Baira', 'Darus Salam', 'East Chandana', 'Farmgate',
    'Khanpur', 'Khulshi', 'Red Crescent Office', 'Sopura', 'Uttar Bagura Road'
]

# Sheet name variants per station (DoE Excel sheets have inconsistent naming)
station_mapping = {
    'Agrabad'            : ['Agrabad'],
    'Baira'              : ['Baira'],
    'Darus Salam'        : ['Darussalam', 'Darus Salam', 'Darussalam '],
    'East Chandana'      : ['East Chandana'],
    'Farmgate'           : ['BARC', 'Farmgate', 'BARC '],
    'Khanpur'            : ['Khanpur'],
    'Khulshi'            : ['Khulshi'],
    'Red Crescent Office': ['Red Crescent Office'],
    'Sopura'             : ['Sopura'],
    'Uttar Bagura Road'  : ['Uttar Bagura Road']
}

# Location file keyword → standard station name mapping
location_keyword_map = {
    'Agrabad'      : 'Agrabad',
    'Baira'        : 'Baira',
    'Darus Salam'  : 'Darus Salam',
    'Chandana'     : 'East Chandana',
    'Farmgate'     : 'Farmgate',
    'Khanpur'      : 'Khanpur',
    'Khulshi'      : 'Khulshi',
    'Red Crescent' : 'Red Crescent Office',
    'Sopura'       : 'Sopura',
    'Bagura'       : 'Uttar Bagura Road'
}


# ==========================================
# 2. HELPER — SMART HEADER HUNTER
#    Now also strips leading empty blocks
#    (fixes Baira-style sheets where real data
#     starts after ~1200 blank rows)
# ==========================================
def extract_sheet(xls, sheet_name):
    """
    1. Finds the real header row (scans up to row 10)
    2. Drops unit rows below the header
    3. Converts types
    4. STRIPS leading empty blocks — rows where ALL
       measurement columns are NaN simultaneously.
       This handles sheets like Baira where ~1200
       blank rows precede the actual data.
    Returns (clean_df, status_message).
    """
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # --- Find real header row ---
    header_found = False
    current_cols = [str(c).strip().lower() for c in df.columns]

    if 'date' in current_cols:
        df.columns = df.columns.astype(str).str.strip()
        header_found = True
    else:
        for i in range(min(10, len(df))):
            row_vals = df.iloc[i].astype(str).str.strip().str.lower().tolist()
            if 'date' in row_vals:
                df.columns = df.iloc[i].astype(str).str.strip()
                df = df.iloc[i + 1:].reset_index(drop=True)
                header_found = True
                break

    if not header_found:
        return None, "No 'Date' column found in first 10 rows"

    # Standardise Date column name
    date_col = next((c for c in df.columns if str(c).strip().lower() == 'date'), None)
    if date_col is None:
        return None, "Date column disappeared after header fix"
    df.rename(columns={date_col: 'Date'}, inplace=True)

    # Drop unit rows (e.g. 'ug/m3' row immediately below headers)
    if len(df) > 0:
        first_row_vals = df.iloc[0].astype(str).str.lower().str.strip().tolist()
        unit_keywords  = ['ug/m3', 'µg/m3', 'mg/m3', '%', 'unit', 'deg', 'w/m2', 'hpa', 'm/s']
        if any(any(kw in val for kw in unit_keywords) for val in first_row_vals):
            df = df.iloc[1:].reset_index(drop=True)

    # Keep only relevant columns
    cols_present = [c for c in ALL_TARGET_COLS if c in df.columns]
    if not cols_present:
        return None, "None of the target variable columns found in sheet"

    df = df[['Date'] + cols_present].copy()

    # Parse types
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    for col in cols_present:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ---- BAIRA FIX ----
    # Drop rows where Date is NaT OR where ALL measurement columns are NaN.
    # This strips leading/trailing empty blocks regardless of how many rows
    # they span — the ~1200 blank rows in Baira disappear here.
    df = df.dropna(subset=['Date'])
    df = df[~df[cols_present].isnull().all(axis=1)].reset_index(drop=True)

    # Filter date range
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    if df.empty:
        return None, f"No valid data rows in range {start_date.date()} to {end_date.date()}"

    return df, "OK"


# ==========================================
# 3. DATA PRESENCE VALIDATOR
#    Checks coverage on the stripped dataframe
#    so leading-empty-block stations aren't
#    penalised on total row count
# ==========================================
def check_data_presence(df):
    """
    Returns (report_dict, all_required_pass).
    Coverage is computed only over rows that
    survived the empty-block strip.
    """
    report   = {}
    all_pass = True
    total    = len(df)

    for var in REQUIRED_VARS:
        if var not in df.columns:
            report[var] = {'column_exists': False, 'non_null_rows': 0, 'coverage_%': 0.0, 'passes': False}
            all_pass = False
        else:
            non_null = int(df[var].notna().sum())
            coverage = round(100 * non_null / total, 1) if total > 0 else 0.0
            passes   = coverage >= 10.0
            report[var] = {'column_exists': True, 'non_null_rows': non_null, 'coverage_%': coverage, 'passes': passes}
            if not passes:
                all_pass = False

    for var in OPTIONAL_VARS:
        if var in df.columns:
            non_null = int(df[var].notna().sum())
            coverage = round(100 * non_null / total, 1) if total > 0 else 0.0
            report[var] = {'column_exists': True, 'non_null_rows': non_null, 'coverage_%': coverage, 'passes': True}
        else:
            report[var] = {'column_exists': False, 'non_null_rows': 0, 'coverage_%': 0.0, 'passes': True}

    return report, all_pass


# ==========================================
# 4. LOAD ALL THREE SOURCE FILES
#    & BUILD SOURCE INVENTORIES
# ==========================================
print("\n" + "="*60)
print("STEP 1: Inventorying all three data sources")
print("="*60)

# --- AOD source ---
print("\n  [AOD] Loading...")
aod_df = pd.read_csv(aod_file)
aod_df = aod_df.rename(columns={'Time': 'Date'})
aod_df['Date'] = pd.to_datetime(aod_df['Date'], dayfirst=True, errors='coerce')
for drop_col in ['month', 'year']:
    if drop_col in aod_df.columns:
        aod_df = aod_df.drop(columns=[drop_col])
aod_df = aod_df[(aod_df['Date'] >= start_date) & (aod_df['Date'] <= end_date)]

aod_stations_in_file = [c for c in aod_df.columns if c != 'Date']
print(f"  [AOD] {len(aod_stations_in_file)} station columns found: {aod_stations_in_file}")

aod_long = aod_df.melt(
    id_vars   =['Date'],
    value_vars = aod_stations_in_file,
    var_name  ='Monitoring_Station',
    value_name='AOD'
)
aod_coverage_map = (
    aod_long.groupby('Monitoring_Station')['AOD']
    .apply(lambda x: round(100 * x.notna().sum() / len(x), 1))
    .to_dict()
)

# --- Location source ---
print("\n  [LOC] Loading...")
loc_df      = pd.read_excel(location_file)
all_loc_stations = loc_df['Station Location'].astype(str).str.strip().tolist()
print(f"  [LOC] {len(all_loc_stations)} stations in location file: {all_loc_stations}")

lat_lon_dict = {}
for _, row in loc_df.iterrows():
    loc_name = str(row['Station Location']).strip()
    for keyword, standard_name in location_keyword_map.items():
        if keyword in loc_name:
            lat_lon_dict[standard_name] = {
                'Latitude' : row['Lattitude'],
                'Longitude': row['Longitude'],
                'Area'     : str(row.get('Area', '')).strip()
            }
            break

loc_clean = pd.DataFrame.from_dict(lat_lon_dict, orient='index').reset_index()
loc_clean = loc_clean.rename(columns={'index': 'Monitoring_Station'})
print(f"  [LOC] Matched {len(loc_clean)} stations to standard names")

# --- DoE source ---
print("\n  [DoE] Loading sheet inventory...")
xls              = pd.ExcelFile(doe_file)
available_sheets = xls.sheet_names
print(f"  [DoE] {len(available_sheets)} sheets found: {available_sheets}")


# ==========================================
# 5. PROCESS DoE SHEETS WITH FULL AUDIT
# ==========================================
print("\n" + "="*60)
print("STEP 2: Extracting & validating DoE station data")
print("="*60)

all_ground_data     = []
selected_stations   = []
completeness_rows   = []   # per-variable detail report
audit_rows          = []   # three-source audit (one row per station)

# Build a lookup: all unique station names across all three sources
all_known_stations = sorted(set(
    aod_stations_in_file +
    list(station_mapping.keys()) +
    list(lat_lon_dict.keys())
))

for station in all_known_stations:

    # --- SOURCE CHECKS ---
    in_aod      = station in aod_stations_in_file
    in_loc      = station in lat_lon_dict
    sheet_opts  = station_mapping.get(station, [])
    valid_sheet = next((s for s in sheet_opts if s in available_sheets), None)
    in_doe      = valid_sheet is not None

    aod_cov = aod_coverage_map.get(station, 0.0)

    audit_row = {
        'Station'              : station,
        'In_AOD_file'          : in_aod,
        'AOD_coverage_%'       : aod_cov if in_aod else 'N/A',
        'In_Location_file'     : in_loc,
        'Latitude'             : lat_lon_dict[station]['Latitude'] if in_loc else 'N/A',
        'Longitude'            : lat_lon_dict[station]['Longitude'] if in_loc else 'N/A',
        'In_DoE_Excel'         : in_doe,
        'DoE_Sheet_Name'       : valid_sheet if in_doe else 'Not found',
        'DoE_Data_Extracted'   : False,
        'Met_Variables_Pass'   : False,
        'Final_Daily_Rows'     : 0,
        'Temporal_Coverage_%'  : 0.0,
        'Selected'             : False,
        'Exclusion_Reason'     : ''
    }

    print(f"\n  [{station}]")
    print(f"    AOD: {'Yes (' + str(aod_cov) + '%)' if in_aod else 'NOT IN AOD FILE'}")
    print(f"    LOC: {'Yes' if in_loc else 'NOT IN LOCATION FILE'}")
    print(f"    DoE: {'Sheet = ' + str(valid_sheet) if in_doe else 'NO MATCHING SHEET'}")

    # Gate 1 — must be in all three sources
    if not in_aod:
        audit_row['Exclusion_Reason'] = 'Not in AOD file — no satellite data available'
        audit_rows.append(audit_row)
        print(f"    -> EXCLUDED: Not in AOD file")
        continue

    if not in_loc:
        audit_row['Exclusion_Reason'] = 'Not in location file — coordinates unavailable'
        audit_rows.append(audit_row)
        print(f"    -> EXCLUDED: Not in location file")
        continue

    if not in_doe:
        audit_row['Exclusion_Reason'] = 'No matching sheet in DoE Excel file'
        audit_rows.append(audit_row)
        print(f"    -> EXCLUDED: No matching DoE sheet")
        continue

    # Gate 2 — extract sheet and check data presence
    df_raw, status = extract_sheet(xls, valid_sheet)

    if df_raw is None:
        audit_row['Exclusion_Reason'] = f'Sheet extraction failed: {status}'
        audit_rows.append(audit_row)
        print(f"    -> EXCLUDED: {status}")
        continue

    audit_row['DoE_Data_Extracted'] = True
    presence, met_pass = check_data_presence(df_raw)

    # Print per-variable detail
    for var, info in presence.items():
        tag  = "REQ" if var in REQUIRED_VARS else "OPT"
        flag = ""
        if var in REQUIRED_VARS:
            flag = "OK" if info['passes'] else "EMPTY — excluded"
        print(f"    {tag} | {var:<20} coverage: {info['coverage_%']:>5}%  ({info['non_null_rows']} rows)  {flag}")

    # Save per-variable detail to completeness report
    comp_row = {'Station': station, 'Total_Rows_After_Strip': len(df_raw)}
    for var, info in presence.items():
        comp_row[f'{var}_coverage_%']  = info['coverage_%']
        comp_row[f'{var}_non_null']    = info['non_null_rows']
        comp_row[f'{var}_col_exists']  = info['column_exists']
    completeness_rows.append(comp_row)

    if not met_pass:
        failed_vars = [v for v in REQUIRED_VARS if not presence[v]['passes']]
        audit_row['Exclusion_Reason'] = f'Required met variables with <10% data: {failed_vars}'
        audit_rows.append(audit_row)
        print(f"    -> EXCLUDED: Insufficient data in {failed_vars}")
        continue

    audit_row['Met_Variables_Pass'] = True

    # Gate 3 — AOD coverage check
    if aod_cov < 10.0:
        audit_row['Exclusion_Reason'] = f'AOD coverage too low: {aod_cov}%'
        audit_rows.append(audit_row)
        print(f"    -> EXCLUDED: AOD coverage too low ({aod_cov}%)")
        continue

    # Aggregate hourly → daily
    agg_dict = {col: 'mean' for col in df_raw.columns if col != 'Date'}
    if 'Rain' in agg_dict:
        agg_dict['Rain'] = 'sum'
    daily_df = df_raw.groupby('Date').agg(agg_dict).reset_index()
    daily_df['Monitoring_Station'] = station

    n_days   = len(daily_df)
    tot_days = (end_date - start_date).days + 1
    temp_cov = round(100 * n_days / tot_days, 1)

    audit_row['Final_Daily_Rows']   = n_days
    audit_row['Temporal_Coverage_%'] = temp_cov

    # Gate 4 — temporal coverage threshold
    if n_days < MIN_DAYS_THRESHOLD:
        audit_row['Exclusion_Reason'] = (
            f'Insufficient temporal coverage: {n_days} days ({temp_cov}%) '
            f'< threshold of {MIN_DAYS_THRESHOLD} days'
        )
        audit_rows.append(audit_row)
        print(f"    -> EXCLUDED: Only {n_days} days ({temp_cov}%) — below {MIN_DAYS_THRESHOLD}-day threshold")
        continue

    # Passed all gates
    audit_row['Selected']          = True
    audit_row['Exclusion_Reason']  = 'Passed all selection criteria'
    audit_rows.append(audit_row)
    all_ground_data.append(daily_df)
    selected_stations.append(station)
    print(f"    -> SELECTED: {n_days} days ({temp_cov}% temporal coverage)")


# ==========================================
# 6. SAVE AUDIT & COMPLETENESS REPORTS
# ==========================================
audit_df      = pd.DataFrame(audit_rows)
completeness_df = pd.DataFrame(completeness_rows)

# Sort: selected first, then by temporal coverage descending
audit_df = audit_df.sort_values(
    ['Selected', 'Final_Daily_Rows'], ascending=[False, False]
).reset_index(drop=True)

audit_df.to_csv(audit_file, index=False)
completeness_df.to_csv(report_file, index=False)

print("\n" + "="*60)
print("STATION SELECTION SUMMARY")
print("="*60)
total_doe_sheets = len(available_sheets)
total_aod_stns   = len(aod_stations_in_file)
total_loc_stns   = len(all_loc_stations)
total_all_known  = len(all_known_stations)

print(f"  DoE Excel sheets total      : {total_doe_sheets}")
print(f"  AOD file stations           : {total_aod_stns}")
print(f"  Location file stations      : {total_loc_stns}")
print(f"  Unique stations (all sources): {total_all_known}")
print(f"")
print(f"  --- Exclusion funnel ---")
not_in_aod  = audit_df[audit_df['In_AOD_file'] == False]
not_in_loc  = audit_df[(audit_df['In_AOD_file'] == True) & (audit_df['In_Location_file'] == False)]
not_in_doe  = audit_df[(audit_df['In_AOD_file'] == True) & (audit_df['In_Location_file'] == True) & (audit_df['In_DoE_Excel'] == False)]
met_fail    = audit_df[(audit_df['DoE_Data_Extracted'] == True) & (audit_df['Met_Variables_Pass'] == False)]
low_cov     = audit_df[(audit_df['Met_Variables_Pass'] == True) & (audit_df['Selected'] == False)]
selected    = audit_df[audit_df['Selected'] == True]

print(f"  Stage 1 — No AOD data            : {len(not_in_aod)} stations → {list(not_in_aod['Station'])}")
print(f"  Stage 2 — No location coords     : {len(not_in_loc)} stations → {list(not_in_loc['Station'])}")
print(f"  Stage 3 — No DoE sheet           : {len(not_in_doe)} stations → {list(not_in_doe['Station'])}")
print(f"  Stage 4 — Met variables empty    : {len(met_fail)} stations → {list(met_fail['Station'])}")
print(f"  Stage 5 — Low temporal coverage  : {len(low_cov)} stations → {list(low_cov['Station'])}")
print(f"  FINAL SELECTED                   : {len(selected)} stations → {list(selected['Station'])}")


# ==========================================
# 7. MERGE EVERYTHING
# ==========================================
if not all_ground_data:
    raise ValueError(
        "CRITICAL: No station passed all selection criteria.\n"
        f"Check '{audit_file}' for exclusion reasons."
    )

print("\n" + "="*60)
print("STEP 3: Merging datasets")
print("="*60)

ground_master = pd.concat(all_ground_data, ignore_index=True)
aod_selected  = aod_long[aod_long['Monitoring_Station'].isin(selected_stations)]

# Inner merge — only rows where both AOD AND ground data exist on the same date
master_df = pd.merge(aod_selected, ground_master, on=['Date', 'Monitoring_Station'], how='inner')
master_df = pd.merge(master_df, loc_clean[['Monitoring_Station', 'Latitude', 'Longitude']], on='Monitoring_Station', how='left')

# Final row-level NaN filter on required columns
required_in_df = [c for c in REQUIRED_VARS if c in master_df.columns]
rows_before    = len(master_df)
master_df      = master_df.dropna(subset=['AOD'] + required_in_df)
rows_dropped   = rows_before - len(master_df)
print(f"  Row-level NaN drop: {rows_dropped} rows removed (scattered missing values)")

# Add season
def get_bd_season(month):
    if pd.isna(month): return np.nan
    if month in [12, 1, 2]:   return 'Winter'
    if month in [3, 4, 5]:    return 'Pre-Monsoon'
    if month in [6, 7, 8, 9]: return 'Monsoon'
    if month in [10, 11]:     return 'Post-Monsoon'
    return 'Unknown'

master_df['Season'] = master_df['Date'].dt.month.apply(get_bd_season)

# Column order
priority_cols = ['Date', 'Monitoring_Station', 'Season', 'Latitude', 'Longitude', 'AOD'] + REQUIRED_VARS + OPTIONAL_VARS
remaining     = [c for c in master_df.columns if c not in priority_cols]
final_cols    = [c for c in priority_cols if c in master_df.columns] + remaining
master_df     = master_df[final_cols]
master_df     = master_df.sort_values(['Monitoring_Station', 'Date']).reset_index(drop=True)
master_df.to_csv(output_file, index=False)


# ==========================================
# 8. FINAL SUMMARY
# ==========================================
end_timer = time.time()

print("\n" + "="*60)
print("FINAL OUTPUT SUMMARY")
print("="*60)
print(f"  Master data     : '{output_file}'")
print(f"  Audit report    : '{audit_file}'")
print(f"  Completeness    : '{report_file}'")
print(f"")
print(f"  Total rows      : {len(master_df)}")
print(f"  Stations        : {master_df['Monitoring_Station'].nunique()} → {list(master_df['Monitoring_Station'].unique())}")
print(f"  Date range      : {master_df['Date'].min().date()} to {master_df['Date'].max().date()}")
print(f"  Columns         : {list(master_df.columns)}")
print(f"")
print(f"  Per-station breakdown:")
print(f"  {'Station':<25} {'Days':>6}  {'Seasons covered'}")
print(f"  {'-'*55}")
for st, grp in master_df.groupby('Monitoring_Station'):
    seasons = ', '.join(sorted(grp['Season'].unique()))
    print(f"  {st:<25} {len(grp):>6}  {seasons}")
print(f"")
print(f"  Time elapsed    : {round(end_timer - start_timer, 2)} seconds")
print("="*60)