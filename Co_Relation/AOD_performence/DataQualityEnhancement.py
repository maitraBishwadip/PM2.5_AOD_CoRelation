import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD THE FEATURE-ENGINEERED DATASET
# ==========================================
input_file = 'Master_Dataset_Feature_Engineered.csv'
output_file = 'Master_Dataset_Final_QC.csv'

print(f"Loading dataset: {input_file}...")
df = pd.read_csv(input_file)
df['Date'] = pd.to_datetime(df['Date'])

# Ensure data is sorted perfectly by Station and Time (CRITICAL for Time-Series Filtering)
df = df.sort_values(by=['Monitoring_Station', 'Date']).reset_index(drop=True)

# ==========================================
# 2. QUALITY CONTROL: PHYSICAL IMPOSSIBILITIES
# ==========================================
print("Applying strict Quality Control (removing negative sensor glitches)...")
# PM2.5 and AOD cannot physically be less than 0. If they are, it's a sensor calibration glitch.
df.loc[df['PM2.5'] < 0, 'PM2.5'] = np.nan
df.loc[df['AOD'] < 0, 'AOD'] = np.nan

# ==========================================
# 3. SHORT-DURATION KALMAN FILTERING (PM2.5 ONLY)
# ==========================================
print("Applying 1D Kalman Filter to ground PM2.5 (Max gap limit: 3 days)...")

def apply_1d_kalman(series):
    """A lightweight 1D Kalman Filter for time-series smoothing and gap filling."""
    n = len(series)
    xhat = np.zeros(n)      # a posteriori estimate
    P = np.zeros(n)         # a posteriori error estimate
    xhatminus = np.zeros(n) # a priori estimate
    Pminus = np.zeros(n)    # a priori error estimate
    K = np.zeros(n)         # gain
    
    # Process variance (how fast we think pollution changes) and Measurement variance (sensor noise)
    Q = 1e-3 
    R = 0.1  
    
    first_valid = series.first_valid_index()
    if first_valid is None: return series # Return if totally empty
    
    # Initialize with first valid measurement
    xhat[0] = series.loc[first_valid]
    P[0] = 1.0
    
    # Convert series to numpy array for fast iteration
    vals = series.values
    
    for k in range(1, n):
        # Time Update (Prediction)
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        
        # Measurement Update (Correction)
        if not np.isnan(vals[k]):
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (vals[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]
        else:
            # If measurement is missing, just use the prediction
            xhat[k] = xhatminus[k]
            P[k] = Pminus[k]
            
    return pd.Series(xhat, index=series.index)

# We process this station by station
qc_dataframes = []

for station, group in df.groupby('Monitoring_Station'):
    # Identify the size of the gaps (consecutive NaNs)
    # This creates a boolean mask where True = valid data, False = NaN
    valid_mask = ~group['PM2.5'].isna()
    
    # This clever trick counts consecutive NaNs
    gap_blocks = valid_mask.cumsum() 
    gap_sizes = group.groupby(gap_blocks)['PM2.5'].transform(lambda x: len(x) - 1)
    
    # Apply the Kalman Filter
    kalman_pm25 = apply_1d_kalman(group['PM2.5'])
    
    # ONLY keep the Kalman imputed values if the original gap was 3 days or less
    # If the gap was > 3 days, we force it back to NaN
    group['PM2.5_QC'] = np.where(gap_sizes <= 3, kalman_pm25, group['PM2.5'])
    
    # We don't want to lose the original data entirely just in case, so we'll keep the QC'd version
    qc_dataframes.append(group)

df_qc = pd.concat(qc_dataframes, ignore_index=True)

# Replace the old PM2.5 with our new Quality-Controlled PM2.5
df_qc['PM2.5'] = df_qc['PM2.5_QC']
df_qc = df_qc.drop(columns=['PM2.5_QC'])

# ==========================================
# 4. THE PROXY GOLDEN RULE
# ==========================================
print("Applying Proxy Golden Rule: Dropping days where Satellite AOD is missing...")
original_length = len(df_qc)
df_qc = df_qc.dropna(subset=['AOD'])
df_qc = df_qc.dropna(subset=['PM2.5']) # Also drop if PM2.5 is still missing after QC
final_length = len(df_qc)

print(f"Dropped {original_length - final_length} rows due to lack of concurrent AOD/PM2.5 readings.")

# ==========================================
# 5. SAVE FINAL DATASET
# ==========================================
df_qc.to_csv(output_file, index=False)
print(f"SUCCESS! Final QC'd dataset saved as '{output_file}'")
print(f"Total concurrent valid days ready for analysis: {final_length}")