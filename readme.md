# AOD as a PM₂.₅ Proxy in Bangladesh
### *Why Aerosol Optical Depth Alone Cannot Predict Surface Air Quality — and What It Takes*

> **Eight years · Ten stations · Four seasons · One conclusion:**
> Raw AOD explains ~2% of PM₂.₅ variance. A physically corrected meteorological system explains >50%.

---

## Table of Contents
- [Background](#background)
- [Dataset](#dataset)
- [Methodology Pipeline](#methodology-pipeline)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [References](#references)

---

## Background

Bangladesh is consistently ranked among the most polluted countries in the world, with annual mean PM₂.₅ concentrations exceeding the WHO guideline of 5 µg/m³ by **fifteen-fold**. Satellite-derived **Aerosol Optical Depth (AOD)** has been widely proposed as a cost-effective surrogate for ground-level PM₂.₅ — particularly in regions with sparse monitoring networks.

However, the physical relationship between column-integrated AOD and surface particle mass is governed by the van Donkelaar (2010) equation:

```
PM₂.₅ = AOD × η
```

where **η** is a conversion factor that depends on:

| Factor | Effect on η |
|---|---|
| Boundary Layer Height (BLH) | Deeper BL dilutes surface PM₂.₅ — same AOD, lower PM₂.₅ |
| Relative Humidity (RH) | Hygroscopic growth inflates AOD optically without increasing dry PM₂.₅ mass |
| Aerosol Type | Continental vs Marine aerosols have different optical–mass relationships |
| Wet Scavenging | Rain removes PM₂.₅ from surface while AOD may remain elevated aloft |

**η is never constant.** In Bangladesh's tropical monsoonal climate, all four components of η vary dramatically across seasons, making raw AOD a fundamentally unreliable proxy. This study proves that empirically.

---

## Dataset

**Period:** January 2014 – December 2021 (8 years)
**Stations:** 10 ground monitoring sites
**Total observations:** 11,045 station-days after QC

| Geo Zone | Stations |
|---|---|
| Coastal | Agrabad, Khulshi |
| Inland Urban | Farmgate, Darus Salam, Khanpur |
| Inland Semi-Urban | Baira, East Chandana, Sopura, Uttar Bagura Road, Red Crescent Office |

**Variables:**

```
Core        : AOD (550 nm), PM₂.₅ (µg/m³)
Meteorology : Temperature, RH, Wind Speed, Wind Direction,
              Solar Radiation, Barometric Pressure, Precipitation
Categorical : Season, Rain_Status, Wind_Origin, Humidity_Profile,
              Temp_Profile, AOD_Loading, Geo_Zone
```

**Seasonal breakdown:**

| Season | Months | N | Mean PM₂.₅ | Mean AOD |
|---|---|---|---|---|
| Winter | Dec–Feb | 3,934 | 141.4 µg/m³ | 0.757 |
| Pre-Monsoon | Mar–May | 2,816 | 82.1 µg/m³ | 0.730 |
| Post-Monsoon | Oct–Nov | 2,670 | 72.5 µg/m³ | 0.576 |
| Monsoon | Jun–Sep | 1,625 | 31.2 µg/m³ | 0.650 |

The 4.5× difference in PM₂.₅ between Winter and Monsoon versus the muted AOD variation is the first visual signal of decoupling.

---

## Methodology Pipeline

```
Raw CSV
   │
   ├─ 1. EDA & Missingness Analysis
   │      ├─ MCAR vs MAR test (KS test per variable)
   │      ├─ Station × Variable heatmap
   │      └─ Season × Variable heatmap
   │
   ├─ 2. Context-Aware Imputation
   │      ├─ Strategy A: Stratified group median      → Temperature, RH, BP
   │      ├─ Strategy B: KNN within Station × Season  → Solar Rad, Wind Speed
   │      └─ Strategy C: Binary + missingness flag    → Wind_Origin
   │
   ├─ 3. Feature Engineering
   │      ├─ BL_proxy      = Temperature / (Wind_Speed + 0.1)
   │      ├─ f_RH          = (1 − RH/100)^0.5
   │      ├─ AOD_FULL_corr = AOD × f_RH / BL_proxy      ← Method 1 correction
   │      ├─ Wet_scavenge  = Rain × AOD
   │      └─ Cyclical encoding for Month, Wind Direction
   │
   ├─ 4. Spearman Correlation Analysis
   │      ├─ Global AOD vs PM₂.₅
   │      └─ Stratified by Season / Rain / Wind / Geo_Zone
   │
   └─ 5. ML Proof Experiment (6 feature groups)
          ├─ G1: Raw AOD only
          ├─ G2: AOD + nonlinear transforms
          ├─ G3: Physically corrected AOD only
          ├─ G4: Met system — NO AOD at all
          ├─ G5: Full corrected system (best)
          └─ Ablation: G5 minus all AOD features
```

### Why global median imputation is wrong here

KS tests confirmed that missing values are **MAR (Missing At Random conditional on group)**, not MCAR. Example:

| Group | Mean RH | Global Median |
|---|---|---|
| Agrabad – Monsoon | 75.4% | 71.3% |
| Darus Salam – Winter BP | 1014.1 hPa | 1009.6 hPa |
| Darus Salam – Monsoon BP | 992.9 hPa | 1009.6 hPa |

Imputing the global median for a Darus Salam–Monsoon BP row introduces a **16 hPa error** — 8× the within-group standard deviation. Stratified imputation reduces this to noise level.

---

## Key Results

### Spearman Correlation

| Variable | ρ with PM₂.₅ | Interpretation |
|---|---|---|
| **Raw AOD** | **+0.137** | **2.3% R² — near-random** |
| Season (ordinal) | −0.660 | Dominant driver |
| Temperature | −0.530 | Strong inverse |
| Barometric Pressure | +0.360 | Stable air mass signal |
| RH | −0.310 | Hygroscopic dilution |
| AOD_FULL_corr | +0.210 | Improved after correction |

**AOD by season:**

| Season | AOD–PM₂.₅ ρ | Proxy quality |
|---|---|---|
| Winter | +0.194 | Poor |
| Post-Monsoon | +0.122 | Poor |
| Monsoon | +0.093 | Poor |
| Pre-Monsoon | **−0.080** | **Actively misleading** |

Pre-Monsoon gives a **negative** correlation — dust intrusion raises AOD while surface PM₂.₅ does not respond proportionally. Using AOD as a proxy here would rank the cleanest days as most polluted.

### ML Proof (Random Forest, temporal holdout 2020–2021)

| Feature Group | R² | What it proves |
|---|---|---|
| G1: Raw AOD only | ~0.02–0.05 | AOD alone is useless |
| G2: AOD + transforms | ~0.05–0.10 | Curve fitting ≠ physics |
| G3: Corrected AOD only | ~0.15–0.25 | Correction alone helps |
| G4: Met system, NO AOD | **>0.50** | **Met beats AOD alone** |
| G5: Full corrected system | Highest | AOD works inside the system |
| Ablation: G5 − AOD | ≈ G5 | AOD's marginal contribution is small |

**Season mean alone (no AOD) → R² = 0.44**. This single number is the core finding: knowing the season explains 22× more PM₂.₅ variance than knowing the AOD value.

### Conditional AOD Performance

AOD-only CV R² by condition (5-fold cross-validation):

```
Best conditions for AOD:
  Winter + No Rain + Continental wind   →  Conditionally usable
  Winter overall                        →  Poor–Usable

Worst conditions:
  Pre-Monsoon (any)                     →  Negative R² (misleading)
  Heavy Rain (any season)               →  Negative R² (misleading)
  Monsoon + Marine wind                 →  Near-zero
```

---

## Repository Structure

```
├── eda_null_analysis.py              # Section 1: EDA + missingness analysis
│                                     # Outputs: EDA_Plots/ (18 figures)
│
├── dataset_creation_correlation.py   # Section 2-3: Imputation + correlation
│                                     # Outputs: Cleaned_Dataset_A.csv
│                                     #          Cleaned_Dataset_B.csv
│                                     #          Correlation_Results.csv
│                                     #          Correlation_Plots/ (10 figures)
│
├── ml_proof_dataset_b.py             # Section 4: ML proof experiment
│                                     # Outputs: ML_B_Results_Summary.csv
│                                     #          ML_B_Feature_Importance.csv
│                                     #          ML_B_Conditional_AOD.csv
│                                     #          ML_Plots_B/ (10 figures)
│
├── Master_Dataset_Final_QC.csv       # Input data (place here before running)
│
└── AOD_PM25_Bangladesh_Report.docx   # Full research report with citations
```

**Generated outputs:**

| File | Description |
|---|---|
| `Cleaned_Dataset_A.csv` | Cleaned only — no feature engineering |
| `Cleaned_Dataset_B.csv` | Cleaned + BL correction + RH correction + interactions |
| `Correlation_Results.csv` | All pairwise Spearman + Pearson with PM₂.₅ and AOD |
| `ML_B_Results_Summary.csv` | All 6 model groups × 2 algorithms × all metrics |
| `ML_B_Feature_Importance.csv` | RF + permutation importance for G5 |
| `ML_B_Conditional_AOD.csv` | AOD-only R² for every condition stratum |

---

## How to Run

**1. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install shap   # optional — for SHAP feature importance
```

**2. Place the input file**
```bash
# Put Master_Dataset_Final_QC.csv in the same directory as the scripts
```

**3. Run in order**
```bash
# Step 1: EDA and missingness analysis
python eda_null_analysis.py

# Step 2: Dataset creation and correlation analysis
python dataset_creation_correlation.py

# Step 3: ML proof experiment
python ml_proof_dataset_b.py
```

Each script auto-creates its output folder in the same directory as the input data. No path configuration needed.

---

## Requirements

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

## References

- van Donkelaar, A., et al. (2010). Global estimates of ambient fine particulate matter from satellite-based AOD. *Environmental Health Perspectives*, 118(6):847–855.
- Levy, R.C., et al. (2007). Second-generation operational algorithm: retrieval of aerosol properties over land from inversion of MODIS spectral reflectance. *JGR Atmospheres*, 112(D13).
- Shahriar, S.A., et al. (2023). Estimating ground-level PM₂.₅ using subset regression and machine learning in Dhaka, Bangladesh. *Air Quality, Atmosphere and Health*, 16(6):1117–1139.
- Rana, M.M., et al. (2022). Spatio-temporal variation of meteorological influence on PM₂.₅ over major urban cities of Bangladesh. *Aerosol and Air Quality Research*, 22:220082.
- Hassan, M.S., et al. (2022). Precipitation is the strongest factor reducing PM₂.₅ in Bangladesh. *Atmospheric Research*, 270:106096.
- Hossen, M.A. and Frey, H.C. (2018). PM₂.₅ and AOD relationships in Bangladesh: seasonal variability. *Atmospheric Environment*, 185:145–157.
- Just, A.C., et al. (2020). Advancing methodologies for applying machine learning to PM₂.₅ spatiotemporal models. *Atmospheric Environment*, 244:117908.
- Lee, M., et al. (2025). Spatiotemporal patterns in air pollution in Dhaka, Bangladesh. *Scientific Reports*, 15:12815.
