import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set academic plot style
sns.set_theme(style="whitegrid")

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
input_file = 'Master_AOD_PM25_TimeSeries_With_Nulls.csv'
output_dir = 'EDA_Visualizations'

# Create a folder to save all the plots
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading Master Dataset...")
df = pd.read_csv(input_file)
df['Date'] = pd.to_datetime(df['Date'])

print("\n" + "="*50)
print("PART 1: THE MISSING DATA STORY")
print("="*50)
# ==========================================
# 2. MISSING DATA ANALYSIS
# ==========================================
# Calculate missing percentages per station
missing_stats = df.groupby('Monitoring_Station').apply(
    lambda x: pd.Series({
        'Total_Days': len(x),
        'Missing_AOD_Count': x['AOD'].isna().sum(),
        'Missing_AOD_%': (x['AOD'].isna().sum() / len(x)) * 100,
        'Missing_PM25_Count': x['PM2.5'].isna().sum(),
        'Missing_PM25_%': (x['PM2.5'].isna().sum() / len(x)) * 100
    })
).reset_index()

print(missing_stats.round(2).to_string(index=False))

# Plot 1: Missing Data Bar Chart
plt.figure(figsize=(12, 6))
missing_stats_melted = missing_stats[['Monitoring_Station', 'Missing_AOD_%', 'Missing_PM25_%']].melt(
    id_vars='Monitoring_Station', var_name='Variable', value_name='Missing Percentage'
)
sns.barplot(data=missing_stats_melted, x='Monitoring_Station', y='Missing Percentage', hue='Variable', palette=['#1f77b4', '#d62728'])
plt.title('Percentage of Missing Data by Station (The Gaps Kalman Must Fill)', fontsize=14, pad=15)
plt.ylabel('Percentage Missing (%)')
plt.xticks(rotation=45)
plt.legend(title='Sensor Type')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_Missing_Data_Percentages.png', dpi=300)
plt.close()

print("\n" + "="*50)
print("PART 2: STATION-WISE DESCRIPTIVE STATISTICS")
print("="*50)
# ==========================================
# 3. SUMMARY STATISTICS & DISTRIBUTIONS
# ==========================================
print("Averages (Means) per station:")
print(df.groupby('Monitoring_Station')[['AOD', 'PM2.5']].mean().round(2))

# Plot 2: Boxplots for Outliers and Distributions
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(data=df, x='Monitoring_Station', y='PM2.5', ax=axes[0], palette='Reds')
axes[0].set_title('PM2.5 Distribution & Outliers by Station', fontsize=12)
axes[0].set_ylabel('PM2.5 Concentration')
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(data=df, x='Monitoring_Station', y='AOD', ax=axes[1], palette='Blues')
axes[1].set_title('AOD Distribution & Outliers by Station', fontsize=12)
axes[1].set_ylabel('Aerosol Optical Depth (AOD)')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{output_dir}/02_Data_Distributions_Boxplots.png', dpi=300)
plt.close()

print("\n" + "="*50)
print("PART 3: SEASONALITY (MONTHLY TRENDS)")
print("="*50)
# ==========================================
# 4. SEASONAL TRENDS
# ==========================================
monthly_avg = df.groupby('Month')[['AOD', 'PM2.5']].mean().reset_index()

# Plot 3: Dual-Axis Line Plot for Seasonality
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Month (1=Jan, 12=Dec)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average PM2.5', color=color, fontsize=12, fontweight='bold')
ax1.plot(monthly_avg['Month'], monthly_avg['PM2.5'], color=color, marker='o', linewidth=2.5, label='PM2.5')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Average AOD', color=color, fontsize=12, fontweight='bold')
ax2.plot(monthly_avg['Month'], monthly_avg['AOD'], color=color, marker='s', linewidth=2.5, linestyle='--', label='AOD')
ax2.tick_params(axis='y', labelcolor=color)
ax2.grid(False)

fig.suptitle('National Monthly Seasonality (AOD vs PM2.5)', fontsize=15, fontweight='bold')
plt.xticks(range(1, 13))
fig.tight_layout()
plt.savefig(f'{output_dir}/03_Monthly_Seasonality_Trend.png', dpi=300)
plt.close()

print("\n" + "="*50)
print("PART 4: 3D SCATTER PLOT")
print("="*50)
# ==========================================
# 5. 3D SCATTER PLOT
# ==========================================
df_clean = df.dropna(subset=['AOD', 'PM2.5']).copy()

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

unique_stations = df_clean['Monitoring_Station'].unique()
station_to_num = {station: i for i, station in enumerate(unique_stations)}
df_clean['Station_Num'] = df_clean['Monitoring_Station'].map(station_to_num)

scatter = ax.scatter(
    df_clean['Station_Num'], 
    df_clean['AOD'], 
    df_clean['PM2.5'], 
    c=df_clean['PM2.5'],     
    cmap='YlOrRd',        
    alpha=0.6,            
    edgecolors='w',       
    s=30                  
)

ax.set_xticks(range(len(unique_stations)))
ax.set_xticklabels(unique_stations, rotation=45, ha='right', fontsize=9)
ax.set_xlabel('Monitoring Station', labelpad=20, fontweight='bold')
ax.set_ylabel('AOD (Aerosol Optical Depth)', labelpad=10, fontweight='bold')
ax.set_zlabel('PM2.5 Concentration', labelpad=10, fontweight='bold')
ax.set_title('3D Distribution: Station vs AOD vs PM2.5', fontsize=16, fontweight='bold')

cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label('PM2.5 Concentration Level', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_3D_Scatter_Location_AOD_PM25.png', dpi=300)
plt.close()

print("\n" + "="*50)
print("PART 5: 2D CONTOUR / DENSITY PLOT")
print("="*50)
# ==========================================
# 6. 2D CONTOUR PLOT (KDE)
# ==========================================
plt.figure(figsize=(10, 8))

# Using seaborn's kdeplot to create a filled contour map of the data density
sns.kdeplot(
    data=df_clean, 
    x='AOD', 
    y='PM2.5', 
    fill=True,          # Fills the contour rings with color
    cmap='mako',        # A beautiful dark-to-light blue/green color map
    thresh=0.05,        # Ignores the extremely sparse 5% outer edges
    levels=12,          # Number of contour "rings"
    alpha=0.8
)

plt.title('2D Contour Plot: Data Density Hotspots (AOD vs PM2.5)', fontsize=16, fontweight='bold')
plt.xlabel('Aerosol Optical Depth (AOD)', fontsize=12, fontweight='bold')
plt.ylabel('PM2.5 Concentration', fontsize=12, fontweight='bold')

# Zoom in slightly to remove the massive outliers that flatten the contour graph
plt.xlim(0, df_clean['AOD'].quantile(0.99))
plt.ylim(0, df_clean['PM2.5'].quantile(0.99))

plt.tight_layout()
plt.savefig(f'{output_dir}/05_Contour_Density_AOD_PM25.png', dpi=300)
plt.close()

print("\n" + "="*50)
print(f"EDA COMPLETE! 5 high-res visualization files are in the '{output_dir}' folder.")
print("="*50 + "\n")