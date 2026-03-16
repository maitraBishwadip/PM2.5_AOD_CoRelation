import pandas as pd
import warnings

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# ==========================================
# 1. FILE PATHS
# ==========================================
aod_file = 'AOD-14-21-daywise.csv'
pm25_file = 'cleaned_PM25_daily_2014_2021.csv'
output_file = 'Master_AOD_PM25_TimeSeries_With_Nulls.csv'

print("Step 1: Building the Master Continuous Calendar...")
# ==========================================
# 2. CREATE CONTINUOUS MASTER CALENDAR
# ==========================================
# Generate every single date from 2014 to 2021
date_range = pd.date_range(start='2014-01-01', end='2021-12-31')

# The exact station names from your AOD file
station_list = [
    'Agrabad', 'Baira', 'Darus Salam', 'East Chandana', 'Farmgate', 
    'Khanpur', 'Khulshi', 'Red Crescent Office', 'Sopura', 'Uttar Bagura Road'
]

# Create a master dataframe with every combination of Date and Station
master_index = pd.MultiIndex.from_product([station_list, date_range], names=['Monitoring_Station', 'Date'])
df_master = pd.DataFrame(index=master_index).reset_index()

# Extract Year and Month for easy grouping later in your EDA
df_master['Year'] = df_master['Date'].dt.year
df_master['Month'] = df_master['Date'].dt.month

print("Step 2: Formatting PM2.5 Data...")
print("Step 2: Formatting and Securing PM2.5 Data...")
# ==========================================
# 3. LOAD, CLEAN, AND DEDUPLICATE PM2.5 DATA
# ==========================================
df_pm25 = pd.read_csv(pm25_file)
df_pm25['Date'] = pd.to_datetime(df_pm25['Date']).dt.normalize()

# CRITICAL FIX: Force strictly 1 row per day per station by averaging any accidental duplicates
df_pm25 = df_pm25.groupby(['Monitoring_Station', 'Date'])['PM2.5'].mean().reset_index()
df_pm25['PM2.5'] = df_pm25['PM2.5'].round(2)

print("Step 3: Formatting, Melting, and Securing AOD Data...")
# ==========================================
# 4. LOAD, MELT, AND DEDUPLICATE AOD DATA
# ==========================================
df_aod = pd.read_csv(aod_file)
df_aod = df_aod.rename(columns={'Time': 'Date'})
df_aod['Date'] = pd.to_datetime(df_aod['Date'], format='%d-%m-%Y', errors='coerce')
df_aod = df_aod.drop(columns=['month', 'year'], errors='ignore')

df_aod_long = df_aod.melt(
    id_vars=['Date'],
    var_name='Monitoring_Station',
    value_name='AOD'
)

# CRITICAL FIX: Force strictly 1 row per day per station by averaging any duplicate satellite passes
df_aod_long = df_aod_long.groupby(['Monitoring_Station', 'Date'])['AOD'].mean().reset_index()
df_aod_long['AOD'] = df_aod_long['AOD'].round(4)

print("Step 4: Merging onto the Calendar (Preserving NaNs)...")
# ==========================================
# 5. MERGE EVERYTHING
# ==========================================
# LEFT JOIN ensures the calendar never breaks. Missing data becomes NaN.
df_merged = pd.merge(df_master, df_aod_long, on=['Monitoring_Station', 'Date'], how='left')
df_merged = pd.merge(df_merged, df_pm25, on=['Monitoring_Station', 'Date'], how='left')

# Reorder columns to look highly professional
df_merged = df_merged[['Monitoring_Station', 'Date', 'Year', 'Month', 'AOD', 'PM2.5']]

# Sort chronologically by Station and Date
df_merged = df_merged.sort_values(by=['Monitoring_Station', 'Date'])

# Save the final masterpiece
df_merged.to_csv(output_file, index=False)

print("\n" + "="*60)
print(f"SUCCESS! Time-Series dataset saved to: {output_file}")
print(f"Total rows (Should be exactly 29,220): {len(df_merged)}")
print(f"Missing AOD values (NaNs preserved for Kalman): {df_merged['AOD'].isna().sum()}")
print(f"Missing PM2.5 values (NaNs preserved for Kalman): {df_merged['PM2.5'].isna().sum()}")
print("="*60 + "\n")

# Show the first 10 rows to prove the NaNs and Dates are correct
print(df_merged.head(10))