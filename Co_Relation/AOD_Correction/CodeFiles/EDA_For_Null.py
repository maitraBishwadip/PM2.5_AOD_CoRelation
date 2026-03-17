"""
=============================================================================
EXPLORATORY DATA ANALYSIS — AOD Proxy Study | Bangladesh Dataset
=============================================================================

PURPOSE:
  Full EDA focused on justifying the null-value handling strategy.
  Answers three questions:

  1. WHAT IS MISSING?
     - Overall missingness per variable
     - Missingness heatmap: Station x Variable
     - Missingness heatmap: Season x Variable
     - Co-occurrence of missingness (which vars go missing together)

  2. IS MISSINGNESS RANDOM? (MCAR vs MAR)
     - KS test: do AOD/PM2.5 distributions differ when each met var
       is present vs absent? If yes → missing is NOT random →
       global median imputation is scientifically wrong

  3. HOW MUCH DO VALUES VARY BY CONTEXT?
     - Global std vs within Station×Season group std
       (reduction % proves stratified imputation is needed)
     - Group median heatmaps: Station × Season for each met variable
     - Boxplots per season across stations

  4. VARIABLE DISTRIBUTIONS & OUTLIERS
     - Histograms + boxplots for all numeric variables
     - Physical outlier flags (BP < 900, Temp > 50, RH > 100)

  5. AOD vs PM2.5 RELATIONSHIP
     - Scatter coloured by Season, Rain_Status, Geo_Zone
     - Spearman rho per season (shows the breakdown)

ALL FIGURES saved as PNG in ./EDA_Plots/

REQUIREMENTS:
  pip install pandas numpy matplotlib seaborn scipy

USAGE:
  Place this file in the same folder as Master_Dataset_Final_QC.csv
  python eda_null_analysis.py
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# All paths are resolved relative to this script's location,
# so outputs always sit next to the dataset regardless of
# where you call the script from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "Master_Dataset_Final_QC.csv")
PLOT_DIR   = os.path.join(SCRIPT_DIR, "EDA_Plots")
DPI        = 150

os.makedirs(PLOT_DIR, exist_ok=True)

SEASON_ORDER  = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
SEASON_COLORS = {
    "Winter"      : "#2166ac",
    "Pre-Monsoon" : "#f4a582",
    "Monsoon"     : "#1a9641",
    "Post-Monsoon": "#d6604d",
}
GEO_COLORS = {
    "Coastal"          : "#1f78b4",
    "Inland_Urban"     : "#e31a1c",
    "Inland_SemiUrban" : "#33a02c",
}
RAIN_COLORS = {
    "No Rain"    : "#2c7bb6",
    "Light Rain" : "#abd9e9",
    "Heavy Rain" : "#d7191c",
}

MET_VARS  = ["Wind Speed", "Temperature", "RH", "Solar Rad", "BP", "Rain"]
MISS_VARS = ["Wind Speed", "Temperature", "RH", "Solar Rad", "BP",
             "Rain", "Wind_Origin", "Wind Dir", "V Wind Speed",
             "Humidity_Profile", "Temp_Profile"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save(fig, filename):
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
df["Month"]     = df["Date"].dt.month
df["Year"]      = df["Date"].dt.year
df["DayOfYear"] = df["Date"].dt.dayofyear

# Keep only cols that exist
MISS_VARS = [v for v in MISS_VARS if v in df.columns]
MET_VARS  = [v for v in MET_VARS  if v in df.columns]

print(f"Shape: {df.shape}")
print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Stations: {sorted(df['Monitoring_Station'].unique())}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
section("SECTION 1: Dataset Overview")

# FIG 1.1 — Observations per station and per season
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Dataset Composition", fontsize=13, fontweight="bold")

# Station counts
station_counts = df["Monitoring_Station"].value_counts()
palette_10 = sns.color_palette("tab10", len(station_counts))
axes[0].barh(station_counts.index, station_counts.values,
             color=palette_10, edgecolor="white")
for i, v in enumerate(station_counts.values):
    axes[0].text(v + 15, i, f"{v:,}", va="center", fontsize=9)
axes[0].set_xlabel("Observations")
axes[0].set_title("Observations per station")
axes[0].set_xlim(0, station_counts.max() * 1.2)

# Season counts
season_counts = df["Season"].value_counts().reindex(SEASON_ORDER)
axes[1].bar(season_counts.index,
            season_counts.values,
            color=[SEASON_COLORS[s] for s in season_counts.index],
            edgecolor="white")
for i, v in enumerate(season_counts.values):
    axes[1].text(i, v + 20, f"{v:,}", ha="center", fontsize=9)
axes[1].set_ylabel("Observations")
axes[1].set_title("Observations per season")
axes[1].set_ylim(0, season_counts.max() * 1.15)
plt.tight_layout()
save(fig, "01_dataset_composition.png")

# FIG 1.2 — Stacked bar: observations per year by season
fig, ax = plt.subplots(figsize=(10, 4))
year_season = (df.groupby(["Year", "Season"])
               .size()
               .unstack(fill_value=0)
               .reindex(columns=SEASON_ORDER, fill_value=0))
bottom = np.zeros(len(year_season))
for s in SEASON_ORDER:
    ax.bar(year_season.index, year_season[s],
           bottom=bottom, color=SEASON_COLORS[s],
           label=s, edgecolor="white", linewidth=0.4)
    bottom += year_season[s].values
ax.set_xlabel("Year")
ax.set_ylabel("Observations")
ax.set_title("Observations per year by season", fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_xticks(year_season.index)
plt.tight_layout()
save(fig, "02_observations_by_year.png")

# FIG 1.3 — Geo_Zone breakdown
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
geo_counts = df["Geo_Zone"].value_counts()
axes[0].pie(geo_counts.values,
            labels=geo_counts.index,
            colors=[GEO_COLORS[g] for g in geo_counts.index],
            autopct="%1.1f%%", startangle=90)
axes[0].set_title("Distribution by Geo Zone")

station_geo = df.groupby(["Monitoring_Station", "Geo_Zone"]).size().unstack(fill_value=0)
station_geo.plot(kind="barh", stacked=True,
                 color=[GEO_COLORS[g] for g in station_geo.columns],
                 ax=axes[1], edgecolor="white")
axes[1].set_title("Station breakdown by Geo Zone")
axes[1].set_xlabel("Observations")
plt.tight_layout()
save(fig, "03_geo_zone_breakdown.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — MISSINGNESS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
section("SECTION 2: Missingness Analysis")

# FIG 2.1 — Overall missingness bar chart
miss_pct = (df[MISS_VARS].isnull().mean() * 100).sort_values(ascending=False)
colors_miss = ["#d73027" if p > 40 else
               "#fc8d59" if p > 15 else
               "#91cf60" for p in miss_pct.values]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(miss_pct.index, miss_pct.values,
               color=colors_miss, edgecolor="white")
ax.axvline(40, color="#d73027", linestyle="--", linewidth=1,
           alpha=0.7, label=">40%: high risk")
ax.axvline(15, color="#fc8d59", linestyle="--", linewidth=1,
           alpha=0.7, label=">15%: moderate risk")
for bar, val in zip(bars, miss_pct.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9)
ax.set_xlabel("% missing")
ax.set_title("Overall missingness per variable", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 108)
plt.tight_layout()
save(fig, "04_overall_missingness.png")

# FIG 2.2 — Missingness heatmap: Station × Variable
miss_station = pd.DataFrame({
    v: df.groupby("Monitoring_Station")[v]
         .apply(lambda x: x.isna().mean() * 100)
    for v in MISS_VARS
})
cmap_miss = LinearSegmentedColormap.from_list(
    "miss", ["#f7f7f7", "#fdd49e", "#fc8d59", "#d7301f"])

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(miss_station.values, aspect="auto",
               cmap=cmap_miss, vmin=0, vmax=100)
ax.set_xticks(range(len(MISS_VARS)))
ax.set_xticklabels(MISS_VARS, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(len(miss_station.index)))
ax.set_yticklabels(miss_station.index, fontsize=9)
for i in range(miss_station.shape[0]):
    for j in range(miss_station.shape[1]):
        val = miss_station.values[i, j]
        c = "white" if val > 60 else "black"
        ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                fontsize=8, color=c)
plt.colorbar(im, ax=ax, label="% missing", shrink=0.8)
ax.set_title("Missingness (%) — Station × Variable\n"
             "(shows missingness is station-specific, not random)",
             fontweight="bold")
plt.tight_layout()
save(fig, "05_missingness_station_variable.png")

# FIG 2.3 — Missingness heatmap: Season × Variable
miss_season = pd.DataFrame({
    v: df.groupby("Season")[v]
         .apply(lambda x: x.isna().mean() * 100)
    for v in MISS_VARS
}).reindex(SEASON_ORDER)

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(miss_season.values, aspect="auto",
               cmap=cmap_miss, vmin=0, vmax=100)
ax.set_xticks(range(len(MISS_VARS)))
ax.set_xticklabels(MISS_VARS, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(4))
ax.set_yticklabels(miss_season.index, fontsize=10)
for i in range(miss_season.shape[0]):
    for j in range(miss_season.shape[1]):
        val = miss_season.values[i, j]
        c = "white" if val > 60 else "black"
        ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                fontsize=9, color=c)
plt.colorbar(im, ax=ax, label="% missing", shrink=0.8)
ax.set_title("Missingness (%) — Season × Variable\n"
             "(Winter has highest missing counts — confirms non-random pattern)",
             fontweight="bold")
plt.tight_layout()
save(fig, "06_missingness_season_variable.png")

# FIG 2.4 — Co-occurrence heatmap (which vars go missing together?)
core_miss = [v for v in
             ["Wind Speed", "Temperature", "RH", "Solar Rad", "BP"]
             if v in df.columns]
miss_indicator = df[core_miss].isnull().astype(int)
co_occur_pct = miss_indicator.T.dot(miss_indicator) / len(df) * 100

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(co_occur_pct, annot=True, fmt=".1f",
            cmap="YlOrRd", ax=ax, linewidths=0.5,
            cbar_kws={"label": "% rows both missing"},
            annot_kws={"size": 10})
ax.set_title("Co-occurrence of missingness (%)\n"
             "(high co-occurrence = same sensor/source going offline together)",
             fontweight="bold")
plt.tight_layout()
save(fig, "07_missingness_cooccurrence.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — IS MISSINGNESS RANDOM? (MCAR vs MAR TEST)
# ═══════════════════════════════════════════════════════════════════════════
section("SECTION 3: MCAR vs MAR Test")

# For each met variable: compare AOD and PM2.5 when it IS vs IS NOT missing
# KS test p < 0.05 → distributions differ → NOT MCAR → global median is wrong

mcar_results = []
test_vars = [v for v in core_miss if v in df.columns]

fig, axes = plt.subplots(2, len(test_vars), figsize=(16, 7))
fig.suptitle(
    "MCAR Test: AOD and PM2.5 distributions when meteorological variable\n"
    "is PRESENT (blue) vs ABSENT (red)  —  p<0.05 means NOT random",
    fontsize=10, fontweight="bold")

for col_i, var in enumerate(test_vars):
    missing_mask = df[var].isna()
    for row_i, target in enumerate(["AOD", "PM2.5"]):
        ax = axes[row_i, col_i]
        present = df.loc[~missing_mask, target].dropna()
        absent  = df.loc[missing_mask, target].dropna()

        ax.hist(present.values, bins=40, alpha=0.55,
                color="#2166ac", density=True,
                label=f"Present (n={len(present):,})")
        ax.hist(absent.values,  bins=40, alpha=0.55,
                color="#d73027", density=True,
                label=f"Absent (n={len(absent):,})")

        if len(absent) > 10:
            ks_stat, ks_p = stats.ks_2samp(present.values, absent.values)
            conclusion = "NOT random ✗" if ks_p < 0.05 else "Random ✓"
            title_color = "#d73027" if ks_p < 0.05 else "#1a9641"
            ax.set_title(
                f"{var}\n{target}: {conclusion}\n"
                f"KS p={ks_p:.3f}",
                fontsize=7.5, color=title_color, fontweight="bold")
            mcar_results.append({
                "Met_Variable"   : var,
                "Target"         : target,
                "KS_statistic"   : round(ks_stat, 4),
                "KS_p_value"     : round(ks_p, 4),
                "Conclusion"     : conclusion,
                "Mean_present"   : round(present.mean(), 3),
                "Mean_absent"    : round(absent.mean(), 3),
                "Mean_difference": round(absent.mean() - present.mean(), 3),
            })
        else:
            ax.set_title(f"{var}\n{target}: n_absent too small", fontsize=7.5)

        if col_i == 0:
            ax.set_ylabel(target, fontsize=9)
        if row_i == 1:
            ax.set_xlabel("Value", fontsize=8)
        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=6.5, loc="upper right")
        ax.tick_params(labelsize=7)

plt.tight_layout()
save(fig, "08_mcar_test.png")

mcar_df = pd.DataFrame(mcar_results)
mcar_csv = os.path.join(PLOT_DIR, "MCAR_test_results.csv")
mcar_df.to_csv(mcar_csv, index=False)
print(f"\n  MCAR Test Results:")
print(mcar_df.to_string(index=False))
print(f"\n  Saved → {mcar_csv}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — WITHIN-GROUP VARIATION
# (proves stratified imputation is better than global median)
# ═══════════════════════════════════════════════════════════════════════════
section("SECTION 4: Within-group variation (justifies stratified imputation)")

strat_vars = [v for v in
              ["Temperature", "RH", "BP", "Solar Rad", "Wind Speed"]
              if v in df.columns]

# FIG 4.1 — Global std vs mean within-group std (Station × Season)
global_std  = df[strat_vars].std()
within_std  = (df.groupby(["Monitoring_Station", "Season"])[strat_vars]
               .std()
               .mean())

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(strat_vars))
w = 0.35
ax.bar(x - w/2, global_std.values, w,
       label="Global std (all data)", color="#d73027", alpha=0.85, edgecolor="white")
ax.bar(x + w/2, within_std.values, w,
       label="Mean within-group std\n(Station × Season)", color="#2166ac",
       alpha=0.85, edgecolor="white")

for i, (g, wv) in enumerate(zip(global_std.values, within_std.values)):
    reduction = (1 - wv / g) * 100 if g > 0 else 0
    ax.text(x[i], max(g, wv) * 1.04,
            f"−{reduction:.0f}%", ha="center", fontsize=9,
            color="#1a9641", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(strat_vars, fontsize=10)
ax.set_ylabel("Standard deviation")
ax.set_title(
    "Global std vs within Station×Season group std\n"
    "Reduction % = variance explained by grouping — "
    "the higher the better for stratified imputation",
    fontweight="bold", fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "09_global_vs_group_std.png")

# FIG 4.2-4.x — Group median heatmaps for each key variable
for var in ["Temperature", "RH", "BP", "Wind Speed"]:
    if var not in df.columns:
        continue
    pivot = (df.groupby(["Monitoring_Station", "Season"])[var]
               .median()
               .unstack()
               .reindex(columns=SEASON_ORDER))
    n_obs = (df.groupby(["Monitoring_Station", "Season"])[var]
               .count()
               .unstack()
               .reindex(columns=SEASON_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left: heatmap of medians
    cmap_var = "RdYlBu_r" if var in ["Temperature", "BP"] else "YlGnBu"
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap=cmap_var,
                ax=axes[0], linewidths=0.5,
                annot_kws={"size": 9},
                cbar_kws={"label": f"Median {var}"})
    axes[0].set_title(
        f"Median {var} — Station × Season\n"
        "(each cell = imputed value for that group)",
        fontweight="bold")
    axes[0].set_xlabel("Season")
    axes[0].set_ylabel("Station")

    # Right: heatmap of observation counts
    sns.heatmap(n_obs, annot=True, fmt=".0f", cmap="Greens",
                ax=axes[1], linewidths=0.5,
                annot_kws={"size": 9},
                cbar_kws={"label": "N observed"})
    axes[1].set_title("N observed per group\n(sample size backing each median)",
                      fontweight="bold")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])

    plt.tight_layout()
    fname = f"10_group_median_{var.replace(' ', '_')}.png"
    save(fig, fname)

# FIG 4.3 — Boxplots per season across stations for Temperature and RH
for var in ["Temperature", "RH"]:
    if var not in df.columns:
        continue
    fig, axes = plt.subplots(1, 4, figsize=(17, 5), sharey=True)
    fig.suptitle(
        f"{var} by station within each season\n"
        "(spread within season shows global median is inadequate)",
        fontweight="bold", fontsize=10)
    for ax, season in zip(axes, SEASON_ORDER):
        sub = df[df["Season"] == season]
        station_order_by_median = (sub.groupby("Monitoring_Station")[var]
                                      .median()
                                      .sort_values()
                                      .index.tolist())
        data_groups = [
            sub.loc[sub["Monitoring_Station"] == s, var].dropna().values
            for s in station_order_by_median
        ]
        bp = ax.boxplot(data_groups, vert=True, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8),
                        flierprops=dict(marker=".", markersize=2, alpha=0.3))
        station_palette = sns.color_palette("tab10", len(station_order_by_median))
        for patch, color in zip(bp["boxes"], station_palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(
            [s.replace(" ", "\n") for s in station_order_by_median],
            fontsize=6.5)
        ax.set_title(season, fontsize=10,
                     color=SEASON_COLORS[season], fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel(var)
    plt.tight_layout()
    save(fig, f"11_boxplot_{var.replace(' ', '_')}_station_season.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — CORRELATION MATRIX
# (justifies KNN feature selection)
# ═══════════════════════════════════════════════════════════════════════════
section("SECTION 5: Spearman correlation matrix")

corr_vars = [v for v in
             ["AOD", "PM2.5", "Temperature", "RH",
              "Wind Speed", "Solar Rad", "BP", "Rain"]
             if v in df.columns]
corr_matrix = df[corr_vars].corr(method="spearman")

fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            annot_kws={"size": 9},
            cbar_kws={"label": "Spearman ρ"})
ax.set_title(
    "Spearman correlation matrix — all numeric variables\n"
    "(used to select KNN predictor features for Solar Rad and Wind Speed imputation)",
    fontweight="bold", fontsize=10)
plt.tight_layout()
save(fig, "12_spearman_correlation_matrix.png")

# Print key correlations for KNN justification
print("\n  Correlations with Solar Rad (KNN predictor selection):")
for var in corr_vars:
    if var == "Solar Rad":
        continue
    mask = df["Solar Rad"].notna() & df[var].notna()
    if mask.sum() > 30:
        rho, p = stats.spearmanr(df.loc[mask, "Solar Rad"],
                                 df.loc[mask, var])
        print(f"    {var:15s}: ρ={rho:+.3f}  p={p:.3e}")

print("\n  Correlations with Wind Speed (KNN predictor selection):")
for var in corr_vars:
    if var == "Wind Speed":
        continue
    mask = df["Wind Speed"].notna() & df[var].notna()
    if mask.sum() > 30:
        rho, p = stats.spearmanr(df.loc[mask, "Wind Speed"],
                                 df.loc[mask, var])
        print(f"    {var:15s}: ρ={rho:+.3f}  p={p:.3e}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — VARIABLE DISTRIBUTIONS & OUTLIERS
# ═══════════════════════════════════════════════════════════════════════════
section("SECTION 6: Variable distributions and outlier detection")

all_num = [v for v in
           ["AOD", "PM2.5", "Wind Speed", "Temperature",
            "RH", "Solar Rad", "BP", "Rain"]
           if v in df.columns]

# FIG 6.1 — Distribution dashboard (histogram + boxplot per variable)
fig, axes = plt.subplots(len(all_num), 2,
                         figsize=(12, len(all_num) * 2.8))
fig.suptitle("Variable distributions — histogram (left) and boxplot (right)",
             fontweight="bold", fontsize=11, y=1.001)

for i, var in enumerate(all_num):
    data = df[var].dropna()

    # Histogram
    axes[i, 0].hist(data.values, bins=60, color="#4393c3",
                    edgecolor="white", linewidth=0.3, alpha=0.85)
    axes[i, 0].set_ylabel(var, fontsize=9)
    axes[i, 0].set_xlabel("Value", fontsize=8)
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    n_outliers = ((data < lower_fence) | (data > upper_fence)).sum()
    axes[i, 0].set_title(
        f"n={len(data):,}  median={data.median():.2f}  "
        f"IQR outliers={n_outliers:,}",
        fontsize=8)

    # Boxplot
    axes[i, 1].boxplot(data.values, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#4393c3", alpha=0.6),
                       medianprops=dict(color="black", linewidth=1.5),
                       flierprops=dict(marker=".", markersize=2, alpha=0.3))
    axes[i, 1].set_ylabel(var, fontsize=9)
    axes[i, 1].set_xlabel("")

plt.tight_layout()
save(fig, "13_variable_distributions.png")

# FIG 6.2 — Physical outlier flags
print("\n  Physical outlier counts:")
outlier_flags = {
    "BP < 900 hPa (sensor error)": (df["BP"] < 900).sum() if "BP" in df.columns else 0,
    "Temperature > 50°C"         : (df["Temperature"] > 50).sum() if "Temperature" in df.columns else 0,
    "RH > 100%"                  : (df["RH"] > 100).sum() if "RH" in df.columns else 0,
    "RH < 0%"                    : (df["RH"] < 0).sum() if "RH" in df.columns else 0,
    "Wind Speed > 50 m/s"        : (df["Wind Speed"] > 50).sum() if "Wind Speed" in df.columns else 0,
    "AOD > 3 (extreme)"          : (df["AOD"] > 3).sum(),
    "PM2.5 > 400 µg/m³"          : (df["PM2.5"] > 400).sum(),
}
for desc, count in outlier_flags.items():
    print(f"    {desc}: {count:,} rows")

fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(list(outlier_flags.keys()),
        list(outlier_flags.values()),
        color=["#d73027" if v > 0 else "#91cf60"
               for v in outlier_flags.values()],
        edgecolor="white")
for i, v in enumerate(outlier_flags.values()):
    ax.text(v + 0.3, i, str(v), va="center", fontsize=9)
ax.set_xlabel("Count of flagged rows")
ax.set_title("Physical outlier flags", fontweight="bold")
ax.set_xlim(0, max(outlier_flags.values()) * 1.2 + 5)
plt.tight_layout()
save(fig, "14_physical_outliers.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — AOD vs PM2.5 RELATIONSHIP
# ═══════════════════════════════════════════════════════════════════════════
section("SECTION 7: AOD vs PM2.5 relationship by condition")

# FIG 7.1 — Scatter by Season (2×2 panel)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("AOD vs PM2.5 by Season\n(Spearman ρ shown per panel)",
             fontweight="bold", fontsize=11)
for ax, season in zip(axes.flat, SEASON_ORDER):
    sub = df[df["Season"] == season].dropna(subset=["AOD", "PM2.5"])
    ax.scatter(sub["AOD"], sub["PM2.5"],
               alpha=0.25, s=8, color=SEASON_COLORS[season])
    rho, p = stats.spearmanr(sub["AOD"], sub["PM2.5"])
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.set_title(f"{season}  (n={len(sub):,})\nρ={rho:+.3f} {stars}",
                 fontsize=10, color=SEASON_COLORS[season], fontweight="bold")
    ax.set_xlabel("AOD")
    ax.set_ylabel("PM2.5 (µg/m³)")
    # Add linear trend line
    if len(sub) > 10:
        z = np.polyfit(sub["AOD"], sub["PM2.5"], 1)
        xline = np.linspace(sub["AOD"].min(), sub["AOD"].max(), 100)
        ax.plot(xline, np.poly1d(z)(xline), "k--", linewidth=1, alpha=0.5)
plt.tight_layout()
save(fig, "15_aod_pm25_by_season.png")

# FIG 7.2 — Scatter by Rain_Status
if "Rain_Status" in df.columns:
    rain_order = ["No Rain", "Light Rain", "Heavy Rain"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("AOD vs PM2.5 by Rain Status\n"
                 "(rain decouples AOD from PM2.5 via wet scavenging)",
                 fontweight="bold")
    for ax, rs in zip(axes, rain_order):
        sub = df[df["Rain_Status"] == rs].dropna(subset=["AOD", "PM2.5"])
        ax.scatter(sub["AOD"], sub["PM2.5"],
                   alpha=0.25, s=8, color=RAIN_COLORS[rs])
        rho, p = stats.spearmanr(sub["AOD"], sub["PM2.5"])
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f"{rs}  (n={len(sub):,})\nρ={rho:+.3f} {stars}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("AOD")
        if ax == axes[0]:
            ax.set_ylabel("PM2.5 (µg/m³)")
    plt.tight_layout()
    save(fig, "16_aod_pm25_by_rain_status.png")

# FIG 7.3 — Scatter by Geo_Zone
if "Geo_Zone" in df.columns:
    geo_zones = df["Geo_Zone"].unique()
    fig, axes = plt.subplots(1, len(geo_zones), figsize=(15, 4), sharey=True)
    fig.suptitle("AOD vs PM2.5 by Geo Zone", fontweight="bold")
    for ax, gz in zip(axes, geo_zones):
        sub = df[df["Geo_Zone"] == gz].dropna(subset=["AOD", "PM2.5"])
        ax.scatter(sub["AOD"], sub["PM2.5"],
                   alpha=0.25, s=8, color=GEO_COLORS.get(gz, "gray"))
        rho, p = stats.spearmanr(sub["AOD"], sub["PM2.5"])
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f"{gz}  (n={len(sub):,})\nρ={rho:+.3f} {stars}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("AOD")
        if ax == axes[0]:
            ax.set_ylabel("PM2.5 (µg/m³)")
    plt.tight_layout()
    save(fig, "17_aod_pm25_by_geo_zone.png")

# FIG 7.4 — PM2.5/AOD proxy ratio distribution by season
df_valid = df[df["AOD"] > 0].copy()
df_valid["proxy_ratio"] = df_valid["PM2.5"] / df_valid["AOD"]
df_valid_clean = df_valid[df_valid["proxy_ratio"] < df_valid["proxy_ratio"].quantile(0.99)]

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
fig.suptitle("PM2.5/AOD proxy ratio distribution by season\n"
             "(high CV = unstable ratio = AOD is a poor proxy in that season)",
             fontweight="bold", fontsize=10)
for ax, season in zip(axes, SEASON_ORDER):
    sub = df_valid_clean[df_valid_clean["Season"] == season]["proxy_ratio"]
    ax.hist(sub.values, bins=50, color=SEASON_COLORS[season],
            edgecolor="white", alpha=0.8)
    cv = sub.std() / sub.mean() * 100 if sub.mean() > 0 else 0
    ax.set_title(f"{season}\nmedian={sub.median():.0f}\nCV={cv:.0f}%",
                 fontsize=9, color=SEASON_COLORS[season], fontweight="bold")
    ax.set_xlabel("PM2.5 / AOD ratio")
    if ax == axes[0]:
        ax.set_ylabel("Count")
plt.tight_layout()
save(fig, "18_proxy_ratio_by_season.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
section("SECTION 8: Summary statistics table")

summary_rows = []
for var in all_num:
    s = df[var].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    n_out = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
    rho, p = stats.spearmanr(df[var].dropna(),
                             df.loc[df[var].notna(), "PM2.5"])
    summary_rows.append({
        "Variable"   : var,
        "N_valid"    : len(s),
        "N_missing"  : int(df[var].isna().sum()),
        "Pct_missing": round(df[var].isna().mean() * 100, 1),
        "Min"        : round(s.min(), 3),
        "Max"        : round(s.max(), 3),
        "Mean"       : round(s.mean(), 3),
        "Median"     : round(s.median(), 3),
        "Std"        : round(s.std(), 3),
        "IQR_outliers": n_out,
        "Spearman_rho_with_PM25": round(rho, 4),
        "Spearman_p" : round(p, 6),
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(PLOT_DIR, "summary_statistics.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\n  Summary statistics table:")
print(summary_df.to_string(index=False))
print(f"\n  Saved → {summary_csv}")

# ═══════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  EDA complete.")
print(f"  All figures saved in: ./{PLOT_DIR}/")
print(f"{'='*65}")
print(f"\n  Files produced:")
for f in sorted(os.listdir(PLOT_DIR)):
    print(f"    {f}")