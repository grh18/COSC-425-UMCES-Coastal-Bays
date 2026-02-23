# UMCES Pigment Data — Linear Model Analysis

## Dataset Description

Oceanographic pigment monitoring data from **18 stations** (stations 1–18) spanning **1999–2014** (16 years).  After cleaning and deduplication, **3,178 observations** were used.

**Response variables analysed:**
- `totchl` — Total Chlorophyll *a* concentration (μg/L)
- `zea_chl` — Zeaxanthin normalised to Chl *a*
- `fuco_chl` — Fucoxanthin normalised to Chl *a*
- `perid_chl` — Peridinin normalised to Chl *a*

**Predictor — pooled models:** Station number (numeric spatial proxy for along-transect position).
**Predictor — per-station models:** Calendar year (temporal trend at each individual site).

---

## Results by Response Variable

### Total Chl a (ug/L)

**Plot:** [`plots/totchl.png`](plots/totchl.png)

#### Pooled Model (X = station)

| Metric | Value |
|--------|-------|
| Slope | `-0.207164` |
| Intercept | `8.429378` |
| R² | `0.0274` |
| p-value | `5.463e-21` (***) |
| n | `3178` |

> **Interpretation:** Significant spatial trend detected (p = 5.46e-21, ***).  Total Chl a (ug/L) **decreases with increasing station number** (R² = 0.027).

#### Per-Station Temporal Trends (X = year)

**2/18** stations show a significant temporal trend (p < 0.05).

- Increasing over time: stations **[11, 12]**

#### Notable Observations

- **Highest mean Total Chl a (ug/L):** Station 4 (mean = 14.4129)
- **Lowest mean Total Chl a (ug/L):** Station 8 (mean = 3.5727)
- **Most variable station (CV):** Station 10

---

### Zeaxanthin / Chl

**Plot:** [`plots/zea_chl.png`](plots/zea_chl.png)

#### Pooled Model (X = station)

| Metric | Value |
|--------|-------|
| Slope | `-0.000991` |
| Intercept | `0.035012` |
| R² | `0.0150` |
| p-value | `4.227e-12` (***) |
| n | `3178` |

> **Interpretation:** Significant spatial trend detected (p = 4.23e-12, ***).  Zeaxanthin / Chl **decreases with increasing station number** (R² = 0.015).

#### Per-Station Temporal Trends (X = year)

**4/18** stations show a significant temporal trend (p < 0.05).

- Increasing over time: stations **[6, 11, 12, 13]**

#### Notable Observations

- **Highest mean Zeaxanthin / Chl:** Station 4 (mean = 0.0661)
- **Lowest mean Zeaxanthin / Chl:** Station 11 (mean = 0.0096)
- **Most variable station (CV):** Station 6

---

### Fucoxanthin / Chl

**Plot:** [`plots/fuco_chl.png`](plots/fuco_chl.png)

#### Pooled Model (X = station)

| Metric | Value |
|--------|-------|
| Slope | `+0.003876` |
| Intercept | `0.249162` |
| R² | `0.0275` |
| p-value | `4.773e-21` (***) |
| n | `3178` |

> **Interpretation:** Significant spatial trend detected (p = 4.77e-21, ***).  Fucoxanthin / Chl **increases with increasing station number** (R² = 0.028).

#### Per-Station Temporal Trends (X = year)

**17/18** stations show a significant temporal trend (p < 0.05).

- Increasing over time: stations **[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]**

#### Notable Observations

- **Highest mean Fucoxanthin / Chl:** Station 11 (mean = 0.3641)
- **Lowest mean Fucoxanthin / Chl:** Station 7 (mean = 0.1880)
- **Most variable station (CV):** Station 7

---

### Peridinin / Chl

**Plot:** [`plots/perid_chl.png`](plots/perid_chl.png)

#### Pooled Model (X = station)

| Metric | Value |
|--------|-------|
| Slope | `-0.000496` |
| Intercept | `0.034656` |
| R² | `0.0030` |
| p-value | `1.886e-03` (**) |
| n | `3178` |

> **Interpretation:** Significant spatial trend detected (p = 1.89e-03, **).  Peridinin / Chl **decreases with increasing station number** (R² = 0.003).

#### Per-Station Temporal Trends (X = year)

**3/18** stations show a significant temporal trend (p < 0.05).

- Increasing over time: stations **[7, 9, 14]**

#### Notable Observations

- **Highest mean Peridinin / Chl:** Station 7 (mean = 0.0753)
- **Lowest mean Peridinin / Chl:** Station 12 (mean = 0.0185)
- **Most variable station (CV):** Station 11

---

## Cross-Pigment Patterns

Stations **[3, 4]** show consistently above-average values across **multiple pigment variables**, suggesting elevated productivity or a distinctive phytoplankton community at these locations.

Significant along-transect spatial gradients (p < 0.05) were detected for: `totchl` (R²=0.027, p=5.46e-21), `zea_chl` (R²=0.015, p=4.23e-12), `fuco_chl` (R²=0.028, p=4.77e-21), `perid_chl` (R²=0.003, p=1.89e-03).  This implies systematic spatial structuring of phytoplankton communities along the monitoring transect.

