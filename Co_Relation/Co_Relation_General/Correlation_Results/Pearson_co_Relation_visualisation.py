import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

output_dir = 'Correlation_Results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 1. LOAD THE EXACT DATA
# ==========================================
data = {
    'Monitoring_Station': [
        'Agrabad', 'Baira', 'Darus Salam', 'East Chandana', 'Farmgate', 
        'Khanpur', 'Khulshi', 'Red Crescent Office', 'Sopura', 'Uttar Bagura Road'
    ],
    'Baseline_R': [0.0548, 0.2098, 0.1558, 0.0380, 0.1481, 0.1143, 0.0853, 0.0846, 0.1328, 0.1338],
    'Kalman_R': [-0.0423, 0.3018, 0.2917, 0.1472, 0.2393, 0.1653, -0.1055, 0.0457, 0.2527, 0.2955]
}

df = pd.DataFrame(data)
df['Improvement'] = df['Kalman_R'] - df['Baseline_R']

# Sort by improvement so the graph naturally flows from worst to best
df = df.sort_values('Improvement', ascending=True).reset_index(drop=True)

# ==========================================
# PLOT 1: THE DUMBBELL / BELL GRAPH (WITH NET TEXT)
# ==========================================
plt.figure(figsize=(12, 8))

# Draw the lines, dots, and net text for each station
for i in range(len(df)):
    start = df['Baseline_R'][i]
    end = df['Kalman_R'][i]
    net = df['Improvement'][i]
    station = df['Monitoring_Station'][i]
    
    # Determine color: Green for improvement, Red for getting worse
    color = '#2ca02c' if net > 0 else '#d62728'
    
    # Draw the main connecting line
    plt.plot([start, end], [i, i], color=color, alpha=0.5, linewidth=4)
    
    # Draw the dots (the "bells" on the barbell)
    plt.scatter(start, i, color='gray', s=100, zorder=3, label='Baseline (r)' if i==0 else "")
    plt.scatter(end, i, color=color, s=100, zorder=3, label='Post-Kalman (r)' if i==0 else "")
    
    # Write the EXACT NET IMPROVEMENT right above the line
    midpoint = (start + end) / 2
    plt.text(midpoint, i + 0.15, f"{net:+.3f}", color=color, fontweight='bold', ha='center', fontsize=10)

# Formatting the graph
plt.yticks(range(len(df)), df['Monitoring_Station'], fontsize=11, fontweight='bold')
plt.axvline(0, color='black', linewidth=1.2, linestyle='--')
plt.xlabel('Pearson Correlation Coefficient (r)', fontweight='bold', fontsize=12)
plt.title('Pearson Correlation Before and After Using the Kalman Filter', fontweight='bold', fontsize=16, pad=15)

# Fix Legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/05_Dumbbell_Net_Improvement.png", dpi=300)
plt.close()

# ==========================================
# PLOT 2: HIGH-RESOLUTION VISUAL TABLE
# ==========================================
fig, ax = plt.subplots(figsize=(12, 5))

# Hide axes completely
ax.axis('tight')
ax.axis('off')

# Format the numbers so they look perfect in the table (4 decimal places)
# Sort alphabetically for the table
df_table = df.sort_values('Monitoring_Station').reset_index(drop=True)
for col in ['Baseline_R', 'Kalman_R', 'Improvement']:
    df_table[col] = df_table[col].apply(lambda x: f"{x:+.4f}")

# Rename columns for the actual display
col_labels = ['Monitoring Station', 'Baseline (r)', 'Post-Kalman (r)', 'Net Improvement']

# Create the table
table = ax.table(
    cellText=df_table[['Monitoring_Station', 'Baseline_R', 'Kalman_R', 'Improvement']].values,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1] 
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)

# Color the header row and make it bold
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50') # Clean dark slate gray header
    else:
        # Alternating row colors for readability
        if row % 2 == 0:
            cell.set_facecolor('#f4f6f9')
            
# Updated Title
plt.title('Pearson Correlation Before and After Using the Kalman Filter', fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(f"{output_dir}/06_Visual_Data_Table_Updated.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"SUCCESS! The Dumbbell Graph (with Net Text) and Updated Table are saved in the '{output_dir}' folder.")