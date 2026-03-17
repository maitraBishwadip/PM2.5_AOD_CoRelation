import pandas as pd
import numpy as np
import time

start_timer = time.time()

# ==========================================
# 1. FILE PATHS & CONFIGURATION
# ==========================================
aod_file = 'AOD-14-21-daywise.csv'
doe_file = 'DoE CAMS Air Qualtiy Data.xlsx'
location_file = 'site_location.xlsx'
output_file = 'Master_Dataset_Daily_Raw.csv'

start_date = pd.to_datetime('2014-01-01')
end_date = pd.to_datetime('2021-12-31')

target_stations = [
    'Agrabad', 'Baira', 'Darus Salam', 'East Chandana', 'Farmgate', 
    'Khanpur', 'Khulshi', 'Red Crescent Office', 'Sopura', 'Uttar Bagura Road'
]

station_mapping = {
    'Agrabad': ['Agrabad'],
    'Baira': ['Baira'],
    'Darus Salam': ['Darussalam', 'Darus Salam', 'Darussalam '],
    'East Chandana': ['East Chandana'],
    'Farmgate': ['BARC', 'Farmgate', 'BARC '],
    'Khanpur': ['Khanpur'],
    'Khulshi': ['Khulshi'],
    'Red Crescent Office': ['Red Crescent Office'],
    'Sopura': ['Sopura'],
    'Uttar Bagura Road': ['Uttar Bagura Road']
}

target_meteo_cols = ['PM2.5', 'Wind Speed', 'Wind Dir', 'Temperature', 'RH', 'Solar Rad', 'BP', 'Rain', 'V Wind Speed']

# ==========================================
# 2. PROCESS AOD DATA 
# ==========================================
print("Processing AOD data...")
aod_df = pd.read_csv(aod_file)

aod_df = aod_df.rename(columns={'Time': 'Date'})
aod_df['Date'] = pd.to_datetime(aod_df['Date'], dayfirst=True, errors='coerce')

if 'month' in aod_df.columns: aod_df = aod_df.drop(columns=['month'])
if 'year' in aod_df.columns: aod_df = aod_df.drop(columns=['year'])

aod_df = aod_df[(aod_df['Date'] >= start_date) & (aod_df['Date'] <= end_date)]

aod_long = aod_df.melt(
    id_vars=['Date'], 
    value_vars=target_stations, 
    var_name='Monitoring_Station', 
    value_name='AOD'
)

# ==========================================
# 3. PROCESS LAT/LON DATA
# ==========================================
print("Processing Location data...")
loc_df = pd.read_excel(location_file)

lat_lon_dict = {}
for idx, row in loc_df.iterrows():
    loc_name = str(row['Station Location']).strip()
    
    standard_name = None
    if 'Agrabad' in loc_name: standard_name = 'Agrabad'
    elif 'Baira' in loc_name: standard_name = 'Baira'
    elif 'Darus Salam' in loc_name: standard_name = 'Darus Salam'
    elif 'Chandana' in loc_name: standard_name = 'East Chandana'
    elif 'Farmgate' in loc_name: standard_name = 'Farmgate'
    elif 'Khanpur' in loc_name: standard_name = 'Khanpur'
    elif 'Khulshi' in loc_name: standard_name = 'Khulshi'
    elif 'Red Crescent' in loc_name: standard_name = 'Red Crescent Office'
    elif 'Sopura' in loc_name: standard_name = 'Sopura'
    elif 'Bagura' in loc_name: standard_name = 'Uttar Bagura Road'
    
    if standard_name:
        lat_lon_dict[standard_name] = {'Latitude': row['Lattitude'], 'Longitude': row['Longitude']}

loc_clean = pd.DataFrame.from_dict(lat_lon_dict, orient='index').reset_index()
loc_clean = loc_clean.rename(columns={'index': 'Monitoring_Station'})

# ==========================================
# 4. PROCESS HOURLY DoE DATA -> DAILY (SMART EXTRACTION)
# ==========================================
print("Processing Hourly DoE CAMS data (This may take a minute)...")
xls = pd.ExcelFile(doe_file)
available_sheets = xls.sheet_names

all_ground_data = []

for station, sheet_options in station_mapping.items():
    valid_sheet = next((s for s in sheet_options if s in available_sheets), None)
    
    if valid_sheet:
        print(f"  - Extracting {station} (Sheet: {valid_sheet})...")
        df = pd.read_excel(xls, sheet_name=valid_sheet)
        
        # --- SMART HEADER HUNTER ---
        header_found = False
        current_cols = [str(c).strip().lower() for c in df.columns]
        
        if 'date' in current_cols:
            df.columns = df.columns.astype(str).str.strip()
            header_found = True
        else:
            # Scan the first 5 rows to find the real headers
            for i in range(min(5, len(df))):
                row_vals = df.iloc[i].astype(str).str.strip().str.lower().tolist()
                if 'date' in row_vals:
                    df.columns = df.iloc[i].astype(str).str.strip() # Set real headers
                    df = df.iloc[i+1:].reset_index(drop=True) # Drop junk above it
                    header_found = True
                    break
                    
        if not header_found:
            print(f"    -> WARNING: Could not find 'Date' column in {station}. Skipping.")
            continue
            
        # Standardize the 'Date' column name (in case it was 'date' or 'DATE')
        date_col_name = next(c for c in df.columns if str(c).strip().lower() == 'date')
        df.rename(columns={date_col_name: 'Date'}, inplace=True)
        
        # Drop the "units" row (e.g., 'ug/m3') if it exists just below the headers
        if len(df) > 0 and 'PM2.5' in df.columns and 'ug/m3' in str(df['PM2.5'].iloc[0]).strip().lower():
            df = df.iloc[1:].reset_index(drop=True)
            
        cols_to_keep = ['Date'] + [col for col in target_meteo_cols if col in df.columns]
        df = df[cols_to_keep]
        
        # Convert Dates and Numbers safely
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        for col in cols_to_keep[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        if df.empty:
            print(f"    -> WARNING: No data found for {station} between 2014-2021.")
            continue
        
        # Aggregation logic
        agg_dict = {col: 'mean' for col in cols_to_keep[1:]}
        if 'Rain' in agg_dict: agg_dict['Rain'] = 'sum'
            
        daily_df = df.groupby('Date').agg(agg_dict).reset_index()
        daily_df['Monitoring_Station'] = station
        all_ground_data.append(daily_df)

if not all_ground_data:
    raise ValueError("CRITICAL ERROR: No data was successfully extracted from any sheet. Check your Excel file structure.")

ground_master = pd.concat(all_ground_data, ignore_index=True)

# ==========================================
# 5. MERGE EVERYTHING & CALCULATE SEASON
# ==========================================
print("Merging AOD, Ground Data, and Locations...")

master_df = pd.merge(aod_long, ground_master, on=['Date', 'Monitoring_Station'], how='outer')
master_df = pd.merge(master_df, loc_clean, on='Monitoring_Station', how='left')

# CALCULATE SEASONS
print("Calculating standard Bangladeshi meteorological seasons...")
def get_bd_season(month):
    if pd.isna(month): return np.nan
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Pre-Monsoon'
    elif month in [6, 7, 8, 9]: return 'Monsoon'
    elif month in [10, 11]: return 'Post-Monsoon'
    return 'Unknown'

master_df['Season'] = master_df['Date'].dt.month.apply(get_bd_season)

master_df = master_df[(master_df['Date'] >= start_date) & (master_df['Date'] <= end_date)]

# Reorder columns
cols = ['Date', 'Monitoring_Station', 'Season', 'Latitude', 'Longitude', 'AOD'] + [c for c in master_df.columns if c not in ['Date', 'Monitoring_Station', 'Season', 'Latitude', 'Longitude', 'AOD']]
master_df = master_df[cols]

master_df = master_df.sort_values(by=['Monitoring_Station', 'Date']).reset_index(drop=True)

master_df.to_csv(output_file, index=False)

end_timer = time.time()
print(f"SUCCESS! Master dataset saved as '{output_file}'")
print(f"Time elapsed: {round(end_timer - start_timer, 2)} seconds")
print(f"Total rows: {len(master_df)}")