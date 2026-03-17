import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD DATASET
# ==========================================
input_file = 'Master_Dataset_Daily_Raw.csv'
output_file = 'Master_Dataset_Feature_Engineered.csv'

print(f"Loading raw dataset: {input_file}...")
df = pd.read_csv(input_file)

# Ensure Date is datetime
# 'format=mixed' tells Pandas to handle inconsistent string lengths
# '.dt.normalize()' strips away the weird hours/milliseconds and resets everything to pure YYYY-MM-DD
df['Date'] = pd.to_datetime(df['Date'], format='mixed').dt.normalize()

# ==========================================
# 2. FEATURE ENGINEERING: SPATIAL ZONE
# ==========================================
print("Engineering feature: Geographic Zones...")
def get_geo_zone(station):
    # Coastal stations are heavily influenced by sea-salt aerosols and marine humidity
    coastal = ['Agrabad', 'Khulshi', 'Baira', 'Uttar Bagura Road']
    # Heavy urban centers with massive localized emissions
    inland_urban = ['Darus Salam', 'Farmgate', 'Khanpur', 'East Chandana']
    # Rest are considered semi-urban/background
    
    if station in coastal: return 'Coastal'
    elif station in inland_urban: return 'Inland_Urban'
    else: return 'Inland_SemiUrban'

df['Geo_Zone'] = df['Monitoring_Station'].apply(get_geo_zone)

# ==========================================
# 3. FEATURE ENGINEERING: WIND ORIGIN (BANGLADESH CONTEXT)
# ==========================================
# North/North-West winds bring transboundary pollution from the Indo-Gangetic Plain
# South/South-East winds bring clean marine air from the Bay of Bengal
print("Engineering feature: Wind Origin (Transboundary vs Marine)...")
def categorize_wind_dir(degree):
    if pd.isna(degree): return np.nan
    # 270 to 360 (NW) and 0 to 90 (NE) -> Continental/Northern
    if (degree >= 270 and degree <= 360) or (degree >= 0 and degree <= 90):
        return 'Continental (Polluted)'
    # 90 to 270 -> Southern/Marine
    elif degree > 90 and degree < 270:
        return 'Marine (Clean)'
    return 'Variable'

df['Wind_Origin'] = df['Wind Dir'].apply(categorize_wind_dir)

# ==========================================
# 4. FEATURE ENGINEERING: METEOROLOGICAL BINS
# ==========================================
print("Engineering feature: Humidity, Temperature, and Rain Profiles...")

# Humidity Profile (Hygroscopic Growth threshold is usually around 70-75%)
df['Humidity_Profile'] = pd.cut(
    df['RH'], 
    bins=[-np.inf, 50, 75, np.inf], 
    labels=['Dry (<50%)', 'Moderate (50-75%)', 'Humid (>75%)']
)

# Temperature Profile (Proxy for Boundary Layer Height)
df['Temp_Profile'] = pd.cut(
    df['Temperature'], 
    bins=[-np.inf, 20, 28, np.inf], 
    labels=['Cool (<20°C)', 'Warm (20-28°C)', 'Hot (>28°C)']
)

# Rain Status (Wet Deposition / Washout effect)
df['Rain_Status'] = pd.cut(
    df['Rain'], 
    bins=[-np.inf, 0.0, 5.0, np.inf], 
    labels=['No Rain', 'Light Rain', 'Heavy Rain']
)

# Aerosol Loading Baseline (Is it a clean day or a heavily polluted day based on AOD?)
df['AOD_Loading'] = pd.cut(
    df['AOD'], 
    bins=[-np.inf, 0.3, 0.8, np.inf], 
    labels=['Low AOD (<0.3)', 'Moderate AOD (0.3-0.8)', 'High AOD (>0.8)']
)

# ==========================================
# 5. REORDER & SAVE
# ==========================================
print("Finalizing dataset structure...")

# Reorder columns to group the new features logically
base_cols = ['Date', 'Monitoring_Station', 'Geo_Zone', 'Latitude', 'Longitude', 'Season']
proxy_cols = ['AOD', 'AOD_Loading', 'PM2.5']
meteo_cols = ['Wind_Origin', 'Humidity_Profile', 'Temp_Profile', 'Rain_Status']
raw_meteo_cols = ['Wind Speed', 'Wind Dir', 'Temperature', 'RH', 'Solar Rad', 'BP', 'Rain', 'V Wind Speed']

# Combine keeping only columns that actually exist in df
final_cols = [c for c in (base_cols + proxy_cols + meteo_cols + raw_meteo_cols) if c in df.columns]

df_final = df[final_cols]

df_final.to_csv(output_file, index=False)
print(f"SUCCESS! Feature-engineered dataset saved as '{output_file}'")
print(f"New features added: Geo_Zone, Wind_Origin, Humidity_Profile, Temp_Profile, Rain_Status, AOD_Loading")