import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
input_file = 'Master_AOD_PM25_TimeSeries_With_Nulls.csv'
output_file = 'Master_AOD_PM25_Kalman_Filtered.csv'

print("Loading dataset...")
df = pd.read_csv(input_file)
df['Date'] = pd.to_datetime(df['Date'])

# ==========================================
# 2. BASELINE CORRELATION (RAW DATA)
# ==========================================
print("Calculating Baseline Pearson Correlation (ignoring NaNs)...")
baseline_corr = df.groupby('Monitoring_Station').apply(
    lambda x: x['AOD'].corr(x['PM2.5'])
).reset_index(name='Baseline_R')

# ==========================================
# 3. APPLYING THE KALMAN FILTER
# ==========================================
print("Initializing Kalman Filter for time-series imputation...")

def apply_kalman_filter(series):
    """
    Applies a 1D Kalman Filter to smooth data and impute missing NaN values.
    """
    # If the entire series is empty, we can't filter it
    if series.isna().all():
        return series
    
    # Mask the NaN values so the Kalman Filter knows where the gaps are
    masked_data = np.ma.masked_invalid(series.values)
    
    # Initialize a basic 1D Random Walk Kalman Filter
    # We use the first valid reading as our starting state
    first_valid = series.dropna().iloc[0]
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=first_valid,
        observation_covariance=1,
        transition_covariance=0.05 # Allows the state to evolve over time
    )
    
    # Run the smoothing algorithm forward and backward over the timeline
    state_means, _ = kf.smooth(masked_data)
    
    # Return the newly filled and smoothed data
    return pd.Series(state_means.flatten(), index=series.index)

# Create new columns for the filtered data to preserve the raw data for comparison
df['AOD_Kalman'] = df.groupby('Monitoring_Station')['AOD'].transform(apply_kalman_filter)
df['PM25_Kalman'] = df.groupby('Monitoring_Station')['PM2.5'].transform(apply_kalman_filter)

# Prevent negative pollution/AOD values (Kalman math can sometimes dip below 0)
df['AOD_Kalman'] = df['AOD_Kalman'].clip(lower=0).round(4)
df['PM25_Kalman'] = df['PM25_Kalman'].clip(lower=0).round(2)

# ==========================================
# 4. POST-KALMAN CORRELATION
# ==========================================
print("Calculating Post-Kalman Pearson Correlation...")
kalman_corr = df.groupby('Monitoring_Station').apply(
    lambda x: x['AOD_Kalman'].corr(x['PM25_Kalman'])
).reset_index(name='Kalman_R')

# ==========================================
# 5. GENERATE COMPARISON TABLE
# ==========================================
# Merge the baseline and kalman results
comparison_df = pd.merge(baseline_corr, kalman_corr, on='Monitoring_Station')

# Calculate how much the Kalman Filter improved (or changed) the correlation
comparison_df['Improvement'] = comparison_df['Kalman_R'] - comparison_df['Baseline_R']

# Formatting for the command line table
comparison_df['Baseline_R'] = comparison_df['Baseline_R'].round(4)
comparison_df['Kalman_R'] = comparison_df['Kalman_R'].round(4)
comparison_df['Improvement'] = comparison_df['Improvement'].round(4)

print("\n" + "="*70)
print(f"{'STATION COMPARISON: BASELINE vs KALMAN FILTERED CORRELATION':^70}")
print("="*70)
print(comparison_df.to_string(index=False))
print("="*70 + "\n")

# ==========================================
# 6. EXPORT THE FINAL MASTERPIECE
# ==========================================
print("Saving the finalized, continuous dataset...")
df.to_csv(output_file, index=False)
print(f"SUCCESS! Filtered dataset stored safely as: {output_file}")