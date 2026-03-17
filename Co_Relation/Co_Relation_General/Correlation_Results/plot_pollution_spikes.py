import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Create output directory
output_dir = 'Spike_Curves'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading Kalman Filtered dataset...")
input_file = 'Master_AOD_PM25_Kalman_Filtered.csv'
df = pd.read_csv(input_file)
df['Date'] = pd.to_datetime(df['Date'])

# Get all unique stations and sort them alphabetically
stations = sorted(df['Monitoring_Station'].unique())

# ==========================================
# CREATE THE CONDENSED 5x2 GRID
# ==========================================
# 5 rows, 2 columns. sharex=True keeps the timeline synced vertically.
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 10), sharex=True)

# Flatten the 5x2 grid into a simple 1D list so we can easily loop through the 10 stations
axes = axes.flatten()

print("Generating condensed side-by-side spike curves...")

for i, station in enumerate(stations):
    ax = axes[i]
    # Isolate and sort data for the specific station
    station_data = df[df['Monitoring_Station'] == station].sort_values('Date')
    
    # Plot the sharp, continuous red spike line
    ax.plot(
        station_data['Date'], 
        station_data['PM25_Kalman'], 
        color='darkred', 
        linewidth=1.2
    )
    
    # Apply the medical/scientific monitor aesthetic
    ax.grid(True, linestyle='--', alpha=0.6, color='#8c92ac') # Soft grid
    ax.set_facecolor('#ffffff') # White background
    
    # Light red border around each individual subplot
    for spine in ax.spines.values():
        spine.set_edgecolor('#ff9999')
        spine.set_linewidth(1.5)
        
    # Station Name as the title inside each box
    ax.set_title(station, loc='left', fontweight='bold', fontsize=11, color='darkred', pad=4)
    
    # Only add the Y-axis label to the far-left column to keep it clean
    if i % 2 == 0:
        ax.set_ylabel('PM2.5', fontweight='bold', fontsize=9)

# Add master title
fig.suptitle('Synchronized PM2.5 Spike Curves (Side-by-Side Comparison)', fontweight='bold', fontsize=18, y=0.98)

# Compress layout so they stack neatly like a dashboard
plt.tight_layout()
# Adjust spacing: top makes room for the title, wspace separates the columns, hspace separates the rows
plt.subplots_adjust(top=0.92, wspace=0.15, hspace=0.35)

# ==========================================
# EXPORT
# ==========================================
output_path = f"{output_dir}/Condensed_SideBySide_Spike_Curves.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"SUCCESS! The condensed 5x2 dashboard has been saved to: {output_path}")