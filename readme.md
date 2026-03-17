# AOD as a PM₂.₅ Proxy in Bangladesh
### *A Multi-Method Empirical Study: Why Aerosol Optical Depth Cannot Predict Surface Air Quality Alone*

> **Study period:** 2014–2021 &nbsp;·&nbsp; **Stations:** 10 &nbsp;·&nbsp; **Observations:** 11,045 station-days
>
> **Core finding in three numbers:**
> Raw AOD R² = **−0.008** &nbsp;|&nbsp; Season mean alone R² = **0.44** &nbsp;|&nbsp; Full corrected system R² = **0.541**
>
> AOD explains essentially zero variance in PM₂.₅. The season you are in explains **55× more**.

---

## Table of Contents

1. [Scientific Background](#1-scientific-background)
2. [Study Area and Monitoring Network](#2-study-area-and-monitoring-network)
3. [Dataset Structure](#3-dataset-structure)
4. [Quality Control Process](#4-quality-control-process)
5. [Missing Value Analysis (MCAR Test)](#5-missing-value-analysis-mcar-test)
6. [Context-Aware Imputation Strategy](#6-context-aware-imputation-strategy)
7. [Feature Engineering](#7-feature-engineering)
8. [Spearman Correlation Analysis](#8-spearman-correlation-analysis)
9. [Machine Learning Proof Experiment](#9-machine-learning-proof-experiment)
10. [Conditional AOD Analysis](#10-conditional-aod-analysis)
11. [Repository Structure and How to Run](#11-repository-structure-and-how-to-run)
12. [References](#12-references)

---

## 1. Scientific Background

Bangladesh is consistently ranked among the most severely air-polluted nations globally.
Annual mean PM₂.₅ concentrations of 79.9 µg/m³ have been recorded — exceeding the WHO
guideline of 5 µg/m³ by **fifteen-fold** and the national standard by more than twice
*(Khandker et al., 2025; WHO, 2021)*. Ground-based monitoring is geographically sparse,
making satellite-derived data attractive as a supplementary tool.

**Aerosol Optical Depth (AOD)** measures the total extinction of solar radiation by
atmospheric aerosols integrated across the full atmospheric column. It has been widely
proposed as a cost-effective surrogate for surface PM₂.₅ concentration, particularly in
data-scarce regions. However, the foundational physical equation governing this relationship
*(van Donkelaar et al., 2010)* is:

```
PM₂.₅ = AOD × η
```

where **η** is a spatially and temporally varying conversion factor that depends on:

| Component of η | Physical mechanism | Effect when ignored |
|---|---|---|
| **Boundary Layer Height (BLH)** | Aerosols in a deep convective BL are diluted over a larger volume — same column AOD yields lower surface PM₂.₅ | AOD overestimates PM₂.₅ in summer/monsoon |
| **Relative Humidity (RH)** | Hygroscopic growth causes particles to absorb water and swell optically, increasing AOD without increasing dry PM₂.₅ mass *(Levy et al., 2007)* | AOD inflated during high-humidity seasons |
| **Aerosol Type** | Continental (anthropogenic) vs Marine (sea-salt) aerosols have fundamentally different mass-extinction relationships | Same AOD → different PM₂.₅ depending on wind origin |
| **Vertical aerosol profile** | AOD above the boundary layer does not contribute to surface PM₂.₅ | Elevated dust and smoke layers inflate AOD without surface signal |

**η is never constant.** In Bangladesh's tropical monsoonal climate all four components change
dramatically across seasons, making raw AOD a fundamentally unreliable proxy. This study
proves that empirically using a rigorous four-stage analytical framework:
EDA → Spearman correlation → ML feature group experiment → conditional analysis.

---

## 2. Study Area and Monitoring Network

The study covers ten air quality monitoring stations across Bangladesh operated under the
Continuous Air Monitoring Station (CAMS) network. The stations span the full diversity of
Bangladesh's pollution environments: coastal, inland urban, and inland semi-urban zones.

### 2.1 Station Inventory

| Station | Geo Zone | Latitude | Longitude | N (obs) | Mean PM₂.₅ (µg/m³) | Mean AOD |
|---|---|---|---|---|---|---|
| **Agrabad** | Coastal | 22.3232°N | 91.8022°E | 1,046 | 85.4 | 0.597 |
| **Baira** | Coastal | 22.8424°N | 89.5398°E | 981 | 84.2 | 0.812 |
| **Darus Salam** | Inland Urban | 23.7809°N | 90.3556°E | 1,300 | 107.3 | 0.742 |
| **East Chandana** | Inland Urban | 23.9550°N | 90.2796°E | 271 | 103.2 | 0.769 |
| **Farmgate** | Inland Urban | 23.7597°N | 90.3894°E | 1,260 | 114.5 | 0.767 |
| **Khanpur** | Inland Urban | 23.6265°N | 90.5077°E | 1,472 | 123.9 | 0.703 |
| **Khulshi** | Coastal | 22.3616°N | 91.7999°E | 763 | 70.1 | 0.610 |
| **Red Crescent Office** | Inland SemiUrban | 24.8889°N | 91.8673°E | 1,253 | 62.5 | 0.536 |
| **Sopura** | Inland SemiUrban | 24.3804°N | 88.6051°E | 1,388 | 83.8 | 0.703 |
| **Uttar Bagura Road** | Coastal | 22.7098°N | 90.3625°E | 1,311 | 89.6 | 0.702 |
| **TOTAL** | — | — | — | **11,045** | **93.4** | **0.691** |

**Key observations:**
- Inland Urban stations (Khanpur, Farmgate, Darus Salam) record the highest PM₂.₅
  (107–124 µg/m³), reflecting dense urban emission sources.
- Coastal stations (Khulshi, Agrabad) record lower PM₂.₅ (70–85 µg/m³) due to sea breeze
  ventilation and cleaner marine air masses.
- Red Crescent Office (semi-urban, northern Bangladesh) has the lowest mean PM₂.₅ (62.5)
  and AOD (0.536) — least industrialised station in the network.
- East Chandana has the fewest observations (271) due to later installation and instrument
  downtime, making it the least statistically representative station.

### 2.2 Geographic Zones

```
Coastal         : Agrabad, Baira, Khulshi, Uttar Bagura Road
                  → Subject to marine air mass influence
                  → Lower PM₂.₅ but sea-salt AOD inflation during monsoon

Inland Urban    : Darus Salam, East Chandana, Farmgate, Khanpur
                  → Dhaka metropolitan area and surroundings
                  → Highest PM₂.₅, most consistent anthropogenic aerosol type

Inland SemiUrban: Red Crescent Office, Sopura
                  → Northern Bangladesh / mixed land use
                  → Intermediate pollution levels, some transboundary influence
```

### 2.3 Seasonal Classification

Following standard Bangladesh meteorological convention *(Rana et al., 2022)*:

| Season | Months | N | Mean PM₂.₅ | Mean AOD | Mean Temp (°C) | Mean RH (%) |
|---|---|---|---|---|---|---|
| **Winter** | Dec–Feb | 3,934 | **141.4 µg/m³** | 0.757 | 21.7 | 68.7 |
| **Pre-Monsoon** | Mar–May | 2,816 | 82.1 µg/m³ | 0.730 | 27.4 | 66.7 |
| **Post-Monsoon** | Oct–Nov | 2,670 | 72.5 µg/m³ | 0.576 | 26.3 | 71.3 |
| **Monsoon** | Jun–Sep | 1,625 | **31.2 µg/m³** | 0.650 | 29.1 | 76.2 |

The **4.5× difference** in PM₂.₅ from Winter to Monsoon versus a narrow AOD range
(0.576–0.757) is the first visual proof of decoupling. PM₂.₅ swings dramatically across
seasons while AOD barely moves — a direct consequence of η varying with BLH, RH, and
aerosol type.

---

## 3. Dataset Structure

### 3.1 Variables

```
Core variables (always present, no missing):
  AOD        — Aerosol Optical Depth at 550 nm
  PM2.5      — Surface fine particulate matter concentration (µg/m³)

Meteorological (partial coverage — see Section 5):
  Temperature   — Air temperature (°C)
  RH            — Relative humidity (%)
  Wind Speed    — Horizontal wind speed (m/s)
  Wind Dir      — Wind direction (0–360°)
  Solar Rad     — Solar radiation (W/m²)
  BP            — Barometric pressure (hPa)
  Rain          — Daily precipitation (mm)
  V Wind Speed  — Vertical wind speed (m/s)  [DROPPED — 95.4% missing]

Categorical (station-recorded):
  Season          — Winter / Pre-Monsoon / Monsoon / Post-Monsoon
  AOD_Loading     — Low (<0.3) / Moderate (0.3–0.8) / High (>0.8)
  Wind_Origin     — Continental (Polluted) / Marine (Clean)
  Humidity_Profile — Dry (<50%) / Moderate (50–75%) / Humid (>75%)
  Temp_Profile    — Cool (<20°C) / Warm (20–28°C) / Hot (>28°C)
  Rain_Status     — No Rain / Light Rain / Heavy Rain

Location:
  Monitoring_Station, Geo_Zone, Latitude, Longitude, Date
```

### 3.2 Dataset Statistics

| Variable | Min | Mean | Median | Max | Std | Missing |
|---|---|---|---|---|---|---|
| AOD | 0.000 | 0.691 | 0.639 | 3.828 | 0.417 | 0 (0%) |
| PM₂.₅ (µg/m³) | 4.08 | 93.4 | 82.2 | 477.5 | 58.8 | 0 (0%) |
| Temperature (°C) | 3.84 | 25.2 | 25.1 | 63.4 | 4.62 | 4,456 (40%) |
| RH (%) | 10.5 | 69.6 | 71.3 | 107.3 | 13.9 | 4,193 (38%) |
| Wind Speed (m/s) | 0.00 | 2.27 | 1.12 | 42.9 | 4.67 | 5,308 (48%) |
| Solar Rad (W/m²) | 0.01 | 221 | 180 | 1018.5 | 169 | 5,457 (49%) |
| BP (hPa) | 0.15* | 1003 | 1010 | 1126* | 29.7 | 5,137 (47%) |
| Rain (mm) | 0.00 | 6.71 | 0.00 | 1613.9* | 53.2 | 94 (1%) |
| V Wind Speed | 0.01 | 0.63 | 0.32 | 3.91 | 0.74 | 10,542 (95%) |

*values marked with asterisk are physical outliers identified and removed in QC

---

## 4. Quality Control Process

Before any analysis, a rigorous QC pipeline was applied to identify and correct
physically impossible sensor readings. This is distinct from imputation of missing values —
QC addresses *wrong* values, not *absent* ones.

### 4.1 Step-by-Step QC Pipeline

```
Step 1: Drop rows with missing AOD or PM₂.₅
        Rationale: These are the two core study variables.
        Any row without them cannot contribute to analysis.
        Rows removed: none (both are 0% missing in the source data)

Step 2: Remove BP sensor errors (BP < 900 hPa)
        Rationale: Normal sea-level BP in Bangladesh is 995–1020 hPa.
        Values below 900 hPa are physically impossible at this
        elevation and represent sensor malfunctions.
        Rows affected: 61 rows → BP set to NaN and later imputed

Step 3: Cap Temperature at 50°C
        Rationale: The maximum recorded temperature in Bangladesh
        history is ~43°C. Values above 50°C are sensor errors.
        Rows affected: 1 row → capped at 50.0°C

Step 4: Clip RH to [0, 100]%
        Rationale: Relative humidity cannot exceed 100% or be negative.
        RH > 100: 3 rows → set to 100.0%
        RH < 0:   0 rows

Step 5: Cap Wind Speed at 50 m/s
        Rationale: Cyclone-force wind. Values above 50 m/s in daily
        averages are instrument errors.
        Rows affected: 0 rows

Step 6: Cap Rain at 3 × 99th percentile (529.5 mm/day)
        Rationale: The 99th percentile of daily rain was 176.5 mm.
        The maximum recorded was 1,613.9 mm — nearly 10× the 99th
        percentile — a clear sensor or logging error.
        Rows affected: 22 rows → capped at 529.5 mm

Step 7: Drop V Wind Speed column entirely
        Rationale: 10,542 out of 11,045 rows (95.4%) are missing.
        The 503 valid observations are insufficient for any reliable
        imputation or analysis. Retaining it would add noise.
        Action: column dropped
```

### 4.2 QC Summary

| Issue | Rows Affected | Action |
|---|---|---|
| BP < 900 hPa (sensor failure) | 61 | Set to NaN → imputed |
| Temperature > 50°C | 1 | Capped at 50.0°C |
| RH > 100% | 3 | Capped at 100.0% |
| Rain > 529.5 mm/day | 22 | Capped at 529.5 mm |
| V Wind Speed (95.4% missing) | 10,542 | Column dropped |
| **Total rows modified** | **87** | **<0.8% of dataset** |

The dataset is largely clean. QC affects fewer than 1% of rows, indicating the source
monitoring network has generally high data integrity. The main challenge is not data
corruption but systematic *absence* of meteorological data at specific stations.

---

## 5. Missing Value Analysis (MCAR Test)

### 5.1 Missing Values Per Station

The most important finding of the EDA is that missing values are **not randomly distributed**.
They are concentrated at specific stations, revealing instrument gaps rather than random sensor
dropout. The table below shows missingness percentage per variable per station:

| Station | Wind Speed | Temperature | RH | Solar Rad | BP | Wind Origin |
|---|---|---|---|---|---|---|
| Agrabad | 21% | 20% | 16% | 15% | 15% | 21% |
| **Baira** | **97%** | **71%** | **74%** | **82%** | **78%** | **96%** |
| Darus Salam | 20% | 25% | 19% | 24% | 27% | 20% |
| **East Chandana** | **100%** | **59%** | **59%** | **92%** | **59%** | **100%** |
| Farmgate | 46% | 38% | 42% | 57% | 64% | 46% |
| **Khanpur** | **50%** | **79%** | **61%** | **89%** | **66%** | **50%** |
| **Khulshi** | **89%** | **45%** | **61%** | **54%** | **58%** | **89%** |
| Red Crescent Office | 17% | 18% | 17% | 17% | 28% | 18% |
| **Sopura** | **82%** | **49%** | **44%** | **79%** | **60%** | **81%** |
| Uttar Bagura Road | 20% | 13% | 13% | 13% | 22% | 20% |

**Critical observations:**
- **East Chandana** has 100% Wind Speed missing and 92% Solar Rad missing — this station
  never had an anemometer or pyranometer installed. Only AOD, PM₂.₅, and a subset of
  temperature variables were recorded.
- **Baira** is missing 97% of Wind Speed and 96% of Wind Origin — effectively no
  meteorological instrumentation beyond basic temperature/RH sensors that are themselves
  71–74% missing.
- **Sopura** and **Khulshi** are missing 79–89% of Wind Speed — consistent with specific
  instrument failure periods spanning multiple years at those sites.
- **Red Crescent Office** and **Uttar Bagura Road** have the best meteorological coverage
  (13–28% missing), suggesting more complete instrumentation.

This station-structured missingness is the reason global imputation fails catastrophically —
see Section 5.2.

### 5.2 Statistical Test: Is Missingness Random? (MCAR vs MAR)

A two-sample Kolmogorov–Smirnov (KS) test was applied to each meteorological variable.
The test asks: **do the distributions of AOD and PM₂.₅ differ significantly between rows
where the variable is present vs absent?**

- If **p > 0.05**: missing is potentially random (MCAR) — simple imputation may be acceptable
- If **p ≤ 0.05**: missing is NOT random (MAR/MNAR) — imputation must respect context

| Variable | Missing | KS stat (AOD) | KS p (AOD) | Random? | Mean PM₂.₅ (present) | Mean PM₂.₅ (absent) | Δ PM₂.₅ |
|---|---|---|---|---|---|---|---|
| Wind Speed | 48.1% | 0.041 | **0.0002** | ❌ NOT random | 96.9 | 89.6 | **−7.3** |
| Temperature | 40.3% | 0.025 | 0.076 | ✓ Random | 95.7 | 90.0 | −5.7 |
| RH | 38.0% | 0.021 | 0.190 | ✓ Random | 96.5 | 88.3 | **−8.3** |
| Solar Rad | 49.4% | 0.024 | 0.076 | ✓ Random | 94.2 | 92.5 | −1.7 |
| BP | 46.5% | 0.031 | **0.012** | ❌ NOT random | 93.7 | 93.0 | −0.7 |

**Key finding:** PM₂.₅ is systematically **lower** when meteorological variables are missing.
For Wind Speed, mean PM₂.₅ drops from 96.9 (observed) to 89.6 (missing) — a 7.3 µg/m³
gap. This means missing rows preferentially correspond to lower-pollution days, likely calm
clear-sky conditions when instruments drop out or fail.

**Why global median is wrong:**
If we impute the global RH median (71.3%) for a missing Agrabad-Monsoon row, we introduce a
systematic error because Agrabad Monsoon mean RH is 75.4% — 4 percentage points off. For
Darus Salam, BP ranges from 992.9 hPa (Monsoon) to 1014.1 hPa (Winter) — a 21 hPa range.
Global BP median (1009.6 hPa) imputed for a Monsoon row would be off by **16.7 hPa**,
eight times the within-group standard deviation of 2.1 hPa.

---

## 6. Context-Aware Imputation Strategy

Because missingness is MAR conditional on station and season, three different imputation
strategies were applied based on the statistical properties of each variable.

### Strategy A — Stratified Group Median (Temperature, RH, BP)

**Used for:** Temperature, RH, BP

**Why:** These variables have tight within-group (Station × Season) distributions. The
group-level median is a reliable representation of the true value for that station in that
season because the standard deviation within a group is far smaller than the global standard
deviation.

**Evidence:**
- Temperature: global std = 4.62°C vs mean within-group std = 2.81°C (39% reduction)
- BP at Agrabad-Winter: mean = 1014.4 hPa, std = **2.1 hPa** — extremely stable
- BP Darus Salam: Monsoon mean = 992.9, Winter mean = 1014.1 — 21 hPa seasonal range
  that global imputation would completely miss

**Fallback chain:** Station × Season → Station only → Season only → global median
This ensures no imputation fails even for the sparsest station-season combinations.

### Strategy B — K-Nearest Neighbour Within Group (Solar Rad, Wind Speed)

**Used for:** Solar Rad, Wind Speed

**Why:** These variables have high within-group variance. Solar Rad at Khulshi has σ = 287 W/m²
within the same station-season group — a group median would smooth away important daily
variation. Instead, KNN (k=5, distance-weighted) searches within the same Station × Season
group and uses the meteorological state of the specific day (Temperature, RH, Rain, Month)
to find the five most similar observed days and interpolate.

**Predictors used for KNN:**
- Temperature, RH, Rain, Month (all fully imputed before this step)

**Fallback:** group median if fewer than 5 neighbours exist in the group.

### Strategy C — Binary Encoding with Missingness Flag (Wind Origin)

**Used for:** Wind_Origin (48.1% missing)

**Why Wind Origin is different from other variables:** Wind Origin is a categorical label
derived from air mass back-trajectory analysis, not a sensor reading. When it is missing, the
trajectory classification was either not run or could not confidently classify the air mass.
This *uncertainty* is itself scientifically informative — it is a distinct meteorological state.

**Encoding:**
```
Continental (Polluted) → Wind_Polluted = 1.0
Marine (Clean)         → Wind_Polluted = 0.0
Missing / Unknown      → Wind_Polluted = 0.5  +  Wind_Origin_missing = 1
```

The missingness flag column enables models to explicitly learn that unknown wind origin
behaves differently from either confirmed Continental or Marine conditions.

### Strategy D — Zero-fill (Rain, 0.9% missing)

**Used for:** Rain (94 rows, all in Winter at inland stations)

**Why:** The 94 missing Rain rows are almost entirely in Winter at stations where dry season
is confirmed by the seasonal pattern. Missing rain records in dry-season Bangladesh are
overwhelmingly absence-of-rain events, not failed measurements. Zero is the physically
correct imputation.

### Imputation QC Validation

After imputation, within-group statistics were compared before and after to confirm that
the imputed values do not distort the distributions:

| Variable | Global Std (before) | Mean within-group Std (after) | Reduction |
|---|---|---|---|
| Temperature | 4.62°C | 2.81°C | 39% |
| RH | 13.91% | 8.45% | 39% |
| BP | 29.7 hPa | 4.8 hPa | 84% |

The BP reduction of 84% confirms that stratified imputation is capturing the dominant
seasonal signal rather than introducing noise.

---

## 7. Feature Engineering

Two datasets were produced:

- **Dataset A** — cleaned and encoded only, no derived features (baseline)
- **Dataset B** — Dataset A plus physical correction features (full system)

### 7.1 Encoding of Categorical Variables

All text columns were converted to numbers the model can learn from. Raw strings such as
`"Monsoon"` or `"Marine (Clean)"` carry no mathematical meaning. Each categorical was
encoded by the nature of its ordering:

| Column | Encoding | Values | Rationale |
|---|---|---|---|
| Season | **Ordinal** | Monsoon=1, Post-Monsoon=2, Pre-Monsoon=3, Winter=4 | Pollution load order — Winter most polluted, Monsoon cleanest |
| AOD_Loading | **Ordinal** | Low=1, Moderate=2, High=3 | Natural magnitude order |
| Rain_Status | **Ordinal** | No Rain=0, Light=1, Heavy=2 | Intensity order |
| Humidity_Profile | **Ordinal** | Dry=1, Moderate=2, Humid=3 | RH level order |
| Temp_Profile | **Ordinal** | Cool=1, Warm=2, Hot=3 | Temperature level order |
| Wind_Origin | **Binary + flag** | Continental=1.0, Marine=0.0, Unknown=0.5 | Two physically distinct states |
| Geo_Zone | **Target encoded** | Replaced by mean PM₂.₅ per zone | Captures spatial pollution signal |
| Monitoring_Station | **Target encoded** | Replaced by mean PM₂.₅ per station | Captures station-level emission baseline |
| Month | **Cyclical** | sin(2π×Month/12), cos(2π×Month/12) | December and January are adjacent |
| Wind Direction | **Cyclical** | sin(deg→rad), cos(deg→rad) | 0° and 360° are the same direction |

### 7.2 Physical Correction Features (Dataset B only)

These features directly approximate the components of η from the van Donkelaar equation:

```python
# Boundary Layer Height proxy
# Higher temperature + lower wind speed = shallower, more stable BL
# = higher surface PM₂.₅ concentration per unit of column AOD
BL_proxy = Temperature / (Wind_Speed + 0.1)   # +0.1 prevents ÷0 on calm days

# Hygroscopic growth factor (Levy et al., 2007)
# f_RH → 1.0 at low humidity (no inflation of AOD)
# f_RH → 0.0 at saturation (fully inflated — AOD and PM₂.₅ completely decoupled)
f_RH = (1 - RH / 100) ** 0.5                  # γ = 0.5 for South Asian aerosols

# Full AOD correction: removes both BLH mismatch and hygroscopic inflation
# This directly approximates η in: PM₂.₅ = AOD × η
AOD_FULL_corr = AOD × f_RH / BL_proxy

# Component corrections (used separately in ML experiment)
AOD_BLH_corr  = AOD / BL_proxy                # BLH correction only
AOD_RH_corr   = AOD × f_RH                    # RH correction only

# Wet scavenging interaction
# Rain removes PM₂.₅ from surface while AOD may remain elevated aloft
Wet_scavenge = Rain × AOD

# Physical interaction terms (meteorological modulation of η)
AOD_x_RH     = AOD × RH           # hygroscopic regime
AOD_x_Temp   = AOD × Temperature  # boundary layer regime
AOD_x_WS     = AOD × Wind_Speed   # dispersion regime
AOD_x_Season = AOD × Season_ord   # seasonal aerosol type
```

---

## 8. Spearman Correlation Analysis

Spearman rank correlation was used rather than Pearson because the AOD–PM₂.₅ relationship
is strongly nonlinear *(Shahriar et al., 2020)*. Spearman ρ measures monotonic association
without assuming linearity, making it more appropriate for bounded, skewed variables
(AOD skewness = 1.32, PM₂.₅ skewness = 0.85).

### 8.1 AOD–PM₂.₅ Global Correlation

| Metric | Value | Interpretation |
|---|---|---|
| Spearman ρ | **+0.137 (\*\*\*)** | Very weak positive monotonic association |
| Pearson r | **+0.152 (\*\*\*)** | Very weak linear association |
| **Pearson R²** | **0.023** | **AOD explains 2.3% of PM₂.₅ variance** |

This is not a weak correlation — it is a near-absence of correlation. For context,
a Spearman ρ of 0.137 is typical of two variables that share a common seasonal cycle
but are otherwise independent.

### 8.2 All Features Ranked by Correlation with PM₂.₅

| Rank | Feature | Spearman ρ | Pearson r | Sig | ρ / ρ_AOD |
|---|---|---|---|---|---|
| 1 | Season_ord | +0.687 | +0.641 | *** | **5.0×** |
| 2 | Temperature | −0.509 | −0.466 | *** | 3.7× |
| 3 | Temp_ord | −0.408 | −0.379 | *** | 3.0× |
| 4 | Station_enc | +0.286 | +0.329 | *** | 2.1× |
| 5 | BP | +0.277 | +0.075 | *** | 2.0× |
| 6 | Month | −0.275 | −0.302 | *** | 2.0× |
| 7 | DayOfYear | −0.267 | −0.297 | *** | 1.9× |
| 8 | GeoZone_enc | +0.257 | +0.298 | *** | 1.9× |
| 9 | WindDir_sin | −0.193 | −0.201 | *** | 1.4× |
| 10 | RH | −0.215 | −0.143 | *** | 1.6× |
| 11 | Humidity_ord | −0.147 | −0.126 | *** | 1.1× |
| 12 | Solar Rad | −0.139 | −0.024 | *** | 1.0× |
| **13** | **Raw AOD** | **+0.137** | **+0.152** | *** | **1.0×** |
| 14 | AOD_Loading_ord | +0.128 | +0.140 | *** | — |
| 15 | Rain | +0.100 | +0.037 | *** | — |
| 16 | Rain_Status_ord | +0.103 | +0.113 | *** | — |
| 17 | Wind_Polluted | +0.002 | +0.018 | ns | — |

Season ordinal alone (ρ = 0.687) is **5× stronger** than AOD (ρ = 0.137) as a predictor
of PM₂.₅. Knowing which station you are at (Station_enc, ρ = 0.286) predicts PM₂.₅ twice
as well as knowing the AOD value. Wind origin has essentially zero correlation with PM₂.₅
(ρ = 0.002, p = 0.85, not significant).

### 8.3 AOD is Independent of its Own Correction Factors

A critical diagnostic: if RH, BP, and BLH are the variables that mediate η, they should
correlate with AOD if AOD were absorbing their signal. They do not:

| Feature | ρ with AOD | Pearson r | Sig |
|---|---|---|---|
| RH | +0.005 | +0.007 | ns |
| BP | −0.004 | −0.003 | ns |
| Temperature | −0.063 | −0.079 | *** |
| Wind_Polluted | +0.028 | +0.024 | ** |

RH and BP — the two dominant modifiers of η — have **no significant correlation with AOD**.
AOD and the physical variables that determine its relationship to PM₂.₅ are largely
orthogonal. This explains precisely why raw AOD fails: it carries none of the meteorological
context needed to convert it to a surface concentration.

---

## 9. Machine Learning Proof Experiment

### 9.1 Design

Six progressive feature groups were tested to isolate the contribution of each component
of the physical system. The key question is not "can we build a good PM₂.₅ model?" but
"what does AOD alone contribute, and when is that contribution meaningful?"

**Train/test split:** 2014–2019 (8,355 rows) → train | 2020–2021 (2,690 rows) → test
Random splitting was rejected: it allows future observations to inform past predictions,
inflating apparent performance. Temporal splitting is the correct approach for time-series
environmental data *(Shahriar et al., 2023)*.

**Models:** Random Forest (primary) and Ridge Regression (linear baseline). RF was chosen
because it captures the nonlinear η relationship without assuming a functional form
*(Chen et al., 2018; Just et al., 2020)*.

| Group | Features included | N features | Scientific purpose |
|---|---|---|---|
| **G1** | Raw AOD only | 1 | Baseline proxy assumption |
| **G2** | AOD + log/sqrt/sq transforms | 4 | Does curve-fitting help without physics? |
| **G3** | AOD_FULL_corr + AOD_BLH_corr + AOD_RH_corr | 3 | Does the η correction alone help? |
| **G4** | Full met system — NO AOD at all | 21 | Does meteorology beat AOD? |
| **G5** | Full Dataset B (all features) | 40 | Best possible: corrected AOD in full system |
| **G5-ablation** | G5 minus ALL AOD features | 26 | What is AOD's marginal contribution? |

### 9.2 Results — Random Forest (Temporal Holdout 2020–2021)

| Group | R² | RMSE (µg/m³) | nRMSE | ΔR² vs G1 | Interpretation |
|---|---|---|---|---|---|
| **G1** Raw AOD | **−0.008** | 64.32 | 0.649 | — | Worse than predicting the mean |
| **G2** AOD transforms | **−0.008** | 64.32 | 0.649 | 0.000 | Transforms add nothing |
| **G3** Corrected AOD | **+0.035** | 62.95 | 0.635 | +0.043 | Correction helps slightly |
| **G4** Met, no AOD | **+0.506** | 45.05 | 0.455 | **+0.514** | Met alone 63× better |
| **G5** Full system | **+0.541** | 43.43 | 0.438 | **+0.549** | Best overall |
| **G5-ablation** | **+0.517** | 44.52 | 0.449 | +0.525 | AOD ΔR² = only +0.024 |

**Linear Ridge confirms the finding is not model-specific:**

| Group | Ridge R² | RMSE |
|---|---|---|
| G1 Raw AOD | +0.012 | 63.68 |
| G3 Corrected AOD | +0.013 | 63.66 |
| G4 No AOD | +0.504 | 45.14 |
| G5 Full system | +0.508 | 44.97 |
| G5-ablation | +0.510 | 44.87 |

**Reading the critical numbers:**

**G1 = −0.008** — the AOD-only RF model is worse than always predicting the global mean
(R² = 0.0 by definition). Negative predictive skill. This is not a poorly tuned model;
it is a correct model telling you that AOD carries no useful information about PM₂.₅ when
used alone.

**G2 = G1 = −0.008** — adding log(AOD), sqrt(AOD), and AOD² makes no difference
whatsoever. The problem is not the functional form of the AOD–PM₂.₅ relationship.
The problem is that there is no relationship to fit.

**G4 = +0.506** — removing AOD entirely and replacing it with meteorological variables
achieves R² = 0.506. The physical met system, with **zero satellite data**, is 63× better
on R² than AOD alone. This is the single most important number in the study.

**G5 vs G5-ablation: ΔR² = 0.024** — adding all AOD features (raw, transformed, and
corrected) to the full system improves R² by just 2.4 percentage points. AOD's marginal
contribution once the physical context is known is negligible.

### 9.3 Feature Importance — What Actually Drives PM₂.₅ Prediction

Top features by permutation importance (measured on unseen test data — more reliable than
impurity-based RF importance):

| Rank | Feature | Category | RF Importance | Permutation Importance |
|---|---|---|---|---|
| 1 | DayOfYear | Temporal | 0.148 | **+0.082** |
| 2 | Season_ord | Temporal | 0.138 | **+0.059** |
| 3 | Month | Temporal | 0.102 | **+0.039** |
| 4 | Month_cos | Temporal | 0.059 | **+0.025** |
| 5 | Month_sin | Temporal | 0.058 | **+0.017** |
| 6 | Temperature | Meteorology | 0.072 | −0.008 |
| 7 | Station_enc | Spatial | 0.060 | +0.001 |
| 8 | GeoZone_enc | Spatial | 0.041 | +0.000 |
| 9 | BP | Meteorology | 0.030 | −0.011 |
| 10 | AOD_x_Season | AOD×Met | 0.027 | −0.001 |
| … | | | | |
| 19 | **AOD_FULL_corr** | Corrected AOD | 0.013 | **−0.010** |
| 35 | **Raw AOD** | RAW_AOD | **0.003** | **−0.001** |
| 40 | AOD_Loading_ord | AOD binned | 0.000 | −0.000 |

**Raw AOD ranks 35th out of 40 features** with permutation importance of −0.001.
Removing it from the model actually improves performance marginally. Even AOD_FULL_corr
(the fully corrected version) ranks 19th with negative permutation importance.

The top five predictors are all **temporal** features — DayOfYear, Season, Month. This
confirms that the dominant driver of PM₂.₅ in Bangladesh is not the atmospheric aerosol
load measured by AOD but the seasonal cycle driven by boundary layer dynamics,
precipitation patterns, and temperature inversions.

**Feature importance by category (summed):**

| Category | Sum RF Importance | Sum Permutation Importance |
|---|---|---|
| **Temporal** (Season, Month, DOY) | **0.376** | **+0.140** |
| **Meteorology** (Temp, RH, WS, etc.) | 0.141 | −0.056 |
| **Spatial** (Station, GeoZone, Lat/Lon) | 0.098 | −0.010 |
| AOD × Met interactions | 0.068 | −0.014 |
| AOD transforms | 0.030 | −0.007 |
| Corrected AOD | 0.027 | −0.028 |
| **Raw AOD** | **0.003** | **−0.001** |

Temporal features account for **37.6% of RF importance** and the only categories with
positive permutation importance. Raw AOD accounts for 0.3% of RF importance — less than
the wind direction cyclical features.

---

## 10. Conditional AOD Analysis

To identify the specific conditions under which AOD carries the most information about PM₂.₅,
an AOD-only Random Forest was trained and evaluated (5-fold cross-validation) within every
condition stratum with at least 30 observations.

### 10.1 Full Conditional Results (sorted best to worst)

| Condition | N | AOD-only CV R² | Spearman ρ | Sig |
|---|---|---|---|---|
| Wind = Unknown | 5,311 | −0.033 | +0.145 | *** |
| Wind = Continental | 1,091 | −0.057 | +0.175 | *** |
| Rain = No Rain | 6,381 | −0.081 | +0.119 | *** |
| Post-Monsoon + No Rain | 1,658 | −0.081 | +0.050 | * |
| Pre-Monsoon + No Rain | 1,513 | −0.122 | −0.072 | ** |
| Monsoon + No Rain | 1,033 | −0.123 | +0.072 | * |
| Season = Monsoon | 1,625 | −0.134 | +0.093 | *** |
| Rain = Heavy Rain | 1,367 | −0.147 | +0.128 | *** |
| Winter + Continental | 486 | −0.200 | +0.129 | ** |
| Season = Pre-Monsoon | 2,816 | −0.200 | −0.080 | *** |
| Post-Monsoon + Heavy Rain | 298 | −0.206 | −0.021 | ns |
| Wind = Marine (Clean) | 4,643 | −0.217 | +0.129 | *** |
| Rain = Light Rain | 3,297 | −0.221 | +0.172 | *** |
| Post-Monsoon + Marine | 1,060 | −0.224 | +0.228 | *** |
| Pre-Monsoon + Marine | 1,281 | −0.254 | −0.132 | *** |
| Pre-Monsoon + Light Rain | 960 | −0.305 | −0.143 | *** |
| Winter + No Rain | 2,177 | −0.315 | +0.150 | *** |
| Post-Monsoon + Continental | 308 | −0.361 | +0.197 | *** |
| Season = Winter | 3,934 | −0.368 | +0.194 | *** |
| Monsoon + Marine | 637 | −0.430 | +0.180 | *** |
| Season = Post-Monsoon | 2,670 | −0.537 | +0.122 | *** |
| Winter + Heavy Rain | 491 | −0.637 | +0.052 | ns |
| Winter + Light Rain | 1,266 | −0.653 | +0.316 | *** |
| Post-Monsoon + Light Rain | 714 | −0.889 | +0.290 | *** |
| Monsoon + Light Rain | 357 | −0.926 | +0.093 | ns |
| Winter + Marine | 1,665 | **−1.099** | +0.286 | *** |
| Monsoon + Continental | 103 | **−1.187** | −0.078 | ns |
| Monsoon + Heavy Rain | 235 | **−1.285** | +0.149 | * |
| Pre-Monsoon + Continental | 194 | **−1.972** | −0.090 | ns |
| Pre-Monsoon + Heavy Rain | 343 | **−1.980** | +0.004 | ns |

### 10.2 Interpretation

**There is no condition where AOD-only achieves positive R².** The "best" result is −0.033
(Wind Unknown). The worst is −1.980 (Pre-Monsoon + Heavy Rain), where the AOD-only model
is **three times worse** than simply predicting the mean PM₂.₅ — using AOD here would
actively mislead.

Key patterns:
- **Winter** is intuitively the "best" season for AOD as a proxy (dry, stable, shallow BL)
  yet AOD-only Winter R² = −0.368. Even under the most favourable physical conditions,
  AOD alone cannot predict PM₂.₅.
- **Pre-Monsoon** gives negative Spearman ρ (−0.080) — dust intrusion raises AOD while
  surface PM₂.₅ does not respond proportionally. The AOD signal is pointing in the wrong
  direction.
- **Winter + Light Rain** (R² = −0.653) is worse than Winter overall (R² = −0.368) —
  precipitation begins decoupling AOD from PM₂.₅ even in the dry season.
- **Continental wind** (R² = −0.057) is the "best" wind condition but still deeply negative.
  The aerosol-type consistency of continental air masses provides no rescue when BLH and RH
  are not corrected for.

---

## 11. Repository Structure and How to Run

### 11.1 File Structure

```
├── eda_null_analysis.py
│     Purpose : Exploratory data analysis + MCAR missingness test
│     Outputs : EDA_Plots/           (18 figures)
│               MCAR_test_results.csv
│               summary_statistics.csv
│
├── dataset_creation_correlation.py
│     Purpose : QC + context-aware imputation + feature engineering
│               + Spearman/Pearson correlation analysis
│     Outputs : Cleaned_Dataset_A.csv          (cleaned, no engineering)
│               Cleaned_Dataset_B.csv          (cleaned + corrections)
│               Correlation_Results.csv        (all features vs PM₂.₅ + AOD)
│               Correlation_Matrix_Spearman.csv
│               Correlation_Matrix_Pearson.csv
│               Correlation_Plots/             (10 figures)
│
├── ml_proof_dataset_b.py
│     Purpose : 6-group ML proof experiment + feature importance
│               + conditional AOD analysis
│     Outputs : ML_B_Results_Summary.csv
│               ML_B_Feature_Importance.csv    (RF + permutation importance)
│               ML_B_Conditional_AOD.csv       (AOD-only R² per condition)
│               ML_Plots_B/                    (10 figures)
│
├── Master_Dataset_Final_QC.csv    ← place input data here
└── AOD_PM25_Bangladesh_Report.docx ← full research report with citations
```

### 11.2 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install shap   # optional — for SHAP feature importance in ML script

# 2. Place your data file in the same folder as the scripts
#    The file must be named: Master_Dataset_Final_QC.csv

# 3. Run in order — each script reads from the previous step's outputs

python eda_null_analysis.py             # ~2 min  — EDA and MCAR test
python dataset_creation_correlation.py  # ~5 min  — imputation and correlation
python ml_proof_dataset_b.py            # ~10 min — ML proof experiment
```

All outputs are written to the same directory as the input CSV. No path configuration
is required — all scripts use `os.path.dirname(os.path.abspath(__file__))` to resolve
paths relative to the script location.

### 11.3 Requirements

```
Python       >= 3.8
pandas       >= 1.3
numpy        >= 1.21
matplotlib   >= 3.4
seaborn      >= 0.11
scipy        >= 1.7
scikit-learn >= 1.0
shap         >= 0.41   (optional)
```

---

## 12. References

| Citation | Key contribution to this study |
|---|---|
| van Donkelaar, A. et al. (2010). *Environ. Health Perspect.* 118(6):847–855 | Foundational PM₂.₅ = AOD × η equation; η component definitions |
| Levy, R.C. et al. (2007). *JGR Atmospheres* 112(D13) | f(RH) = (1−RH/100)^0.5 correction; γ = 0.5 for South Asian aerosols |
| van Donkelaar, A. et al. (2006). *JGR Atmospheres* 111(D21) | Vertical aerosol profile as dominant η factor; reduces r from 0.69 to 0.36 if ignored |
| Shahriar, S.A. et al. (2023). *Air Qual. Atmos. Health* 16(6):1117–1139 | Bangladesh ML baseline; temporal splitting methodology |
| Shahriar, S.A. et al. (2020). *Discover Applied Sciences* 2:729 | Univariate AOD R² < 0.1; multivariate raises R² > 0.6 in Bangladesh |
| Rana, M.M. et al. (2022). *Aerosol Air Qual. Res.* 22:220082 | Meteorological drivers of PM₂.₅ variability in Bangladesh; seasonal classification |
| Hassan, M.S. et al. (2022). *Atmos. Res.* 270:106096 | Precipitation as dominant PM₂.₅ scavenger; Bangladesh-specific evidence |
| Hossen, M.A. & Frey, H.C. (2018). *Atmos. Environ.* 185:145–157 | Wind origin and seasonal AOD–PM₂.₅ coupling in Bangladesh |
| Lee, M. et al. (2025). *Scientific Reports* 15:12815 | PM₂.₅ = 237 µg/m³ January vs 107 µg/m³ July in Dhaka — seasonal magnitude |
| Just, A.C. et al. (2020). *Atmos. Environ.* 244:117908 | AOD ranks low in multi-variable RF; temperature and RH outrank AOD |
| Li, Z. et al. (2018). *Natl. Sci. Rev.* 4(6):810–833 | BL–aerosol–PM₂.₅ physical interaction mechanisms |
| Khandker, R. et al. (2025). *Research Square* Preprint | Bangladesh AOD trend 0.49 (2001) → 0.87 (2024), R² = 0.83 |
| WHO (2021). *Global Air Quality Guidelines* | PM₂.₅ annual guideline 5 µg/m³ |
| Chen, Z. et al. (2018). *Sci. Total Environ.* 636:52–60 | RF for PM₂.₅ estimation in China; methodology reference |

---

