import pandas as pd
import warnings
import time

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# ==========================================
# 1. FILE PATHS
# ==========================================
doe_file_path = 'DoE CAMS Air Qualtiy Data.xlsx'
output_pm25_file = 'cleaned_PM25_daily_2014_2021.csv'

# ==========================================
# 2. STATION NAME MAPPING DICTIONARY (With Fallbacks)
# ==========================================
# Keys = AOD Name | Values = List of possible Excel Tab Names
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

start_date = pd.to_datetime('2014-01-01')
end_date = pd.to_datetime('2021-12-31')

pm25_daily_frames = []

print("Opening Excel file pointer (Optimized Auto-Hunting Mode)...")
start_time = time.time()

# ==========================================
# 3. OPTIMIZED DATA EXTRACTION
# ==========================================
with pd.ExcelFile(doe_file_path) as xls:
    available_sheets = xls.sheet_names
    print(f"\n[DEBUG] Tabs found in Excel file: {available_sheets}\n")
    
    for aod_name, possible_tabs in station_mapping.items():
        # Find the first matching tab name from our list of possibilities
        tab_name = next((t for t in possible_tabs if t in available_sheets), None)
        
        if not tab_name:
            print(f"[!] Skipping {aod_name}: None of the expected tab names {possible_tabs} were found in the file.\n")
            continue
            
        print(f"Reading sheet: [{tab_name}] from disk...")
        
        # Read the sheet WITHOUT assuming the first row is the header
        df_sheet = pd.read_excel(xls, sheet_name=tab_name, header=None)
        
        # --- THE HEADER HUNTER ---
        header_row_idx = -1
        for i in range(min(20, len(df_sheet))):  # Search the first 20 rows
            # Convert the row to text and make it lowercase to check
            row_values = df_sheet.iloc[i].astype(str).str.lower().tolist()
            
            # If we see 'date' and 'pm2.5' in this exact row, we found our true headers!
            if 'date' in row_values and any('pm2.5' in val for val in row_values):
                header_row_idx = i
                break
        
        if header_row_idx == -1:
            print(f"  [!] Skipping {tab_name}: Could not locate the 'Date' and 'PM2.5' headers in the first 20 rows.\n")
            continue
            
        # Set the actual headers and drop all the junk rows above it
        df_sheet.columns = df_sheet.iloc[header_row_idx].astype(str).str.strip()
        df_sheet = df_sheet.iloc[header_row_idx + 1:].reset_index(drop=True)
        
        # Now find the exact columns dynamically
        date_col = next((col for col in df_sheet.columns if col.lower() == 'date'), None)
        pm25_col = next((col for col in df_sheet.columns if 'pm2.5' in col.lower()), None)
        
        if date_col and pm25_col:
            # Isolate ONLY the two columns we care about
            df_clean = df_sheet[[date_col, pm25_col]].copy()
            df_clean = df_clean.rename(columns={date_col: 'Date', pm25_col: 'PM2.5'})
            
            # Convert to correct data types
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            df_clean['PM2.5'] = pd.to_numeric(df_clean['PM2.5'], errors='coerce')
            
            # Filter strictly for the 2014-2021 timeframe
            df_clean = df_clean[(df_clean['Date'] >= start_date) & (df_clean['Date'] <= end_date)]
            
            # Group by Date to get the 24-hour daily average
            daily_avg = df_clean.groupby('Date')['PM2.5'].mean().reset_index()
            daily_avg['Monitoring_Station'] = aod_name
            
            # Drop any days where PM2.5 data was completely missing
            daily_avg = daily_avg.dropna(subset=['PM2.5'])
            
            pm25_daily_frames.append(daily_avg)
            print(f"  [+] Extracted {len(daily_avg)} valid daily records for {aod_name}.\n")
            
            # Delete the raw sheets from memory
            del df_sheet, df_clean
        else:
            print(f"  [!] Failed to extract after finding headers. Found: {df_sheet.columns.tolist()}\n")

# ==========================================
# 4. COMPILE AND EXPORT
# ==========================================
if pm25_daily_frames:
    print("Combining into a single concise CSV...")
    master_pm25_df = pd.concat(pm25_daily_frames, ignore_index=True)
    
    # Reorder columns to look professional
    master_pm25_df = master_pm25_df[['Monitoring_Station', 'Date', 'PM2.5']]
    
    # Sort chronologically by Station and Date
    master_pm25_df = master_pm25_df.sort_values(by=['Monitoring_Station', 'Date'])
    
    # Save to CSV
    master_pm25_df.to_csv(output_pm25_file, index=False)
    
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    
    print("="*60)
    print(f"SUCCESS! Clean PM2.5 dataset saved to: {output_pm25_file}")
    print(f"Total valid daily records extracted: {len(master_pm25_df)}")
    print(f"Total processing time: {elapsed_time} seconds")
    print("="*60 + "\n")
else:
    print("Error: No data was successfully extracted.")