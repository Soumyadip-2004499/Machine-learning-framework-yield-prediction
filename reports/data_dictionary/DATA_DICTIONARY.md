# KrishiYield: Comprehensive Agricultural Data Dictionary & Research Mapping

## 1. Executive Summary & Provenance

The **KrishiYield** platform utilizes two benchmark agricultural datasets spanning **39 years (1980–2018)** across **105 major agricultural counties** in 13 states of the United States Corn and Soybean Belt (Illinois, Iowa, Indiana, Ohio, Minnesota, Missouri, Nebraska, South Dakota, North Dakota, Kansas, Wisconsin, Michigan, Kentucky).

### Core Literature Foundation
1. **Iniyan, S., & Jebakumar, R. (2021/2022).** *"Mutual Information Feature Selection (MIFS) Based Crop Yield Prediction on Corn and Soybean Crops Using Multilayer Stacked Ensemble Regression (MSER)."* *Wireless Personal Communications*, 126(3), 1935–1964. [DOI: 10.1007/s11277-021-08712-9](https://doi.org/10.1007/s11277-021-08712-9)
2. **Khaki, S., Wang, L., & Archontoulis, S. V. (2020).** *"A CNN–RNN Framework for Crop Yield Prediction."* *Frontiers in Plant Science*, 10, 1750.
3. **Waqar, M., Kim, Y.-W., & Byun, Y.-C. (2025).** *"A Stacking Ensemble Framework Leveraging Synthetic Data for Accurate and Stable Crop Yield Forecasting."* *IEEE Access*, 13, 136909–136926.
4. **SRM IST Project Paper (2026).** *"Machine Learning Framework For Yield Prediction Using Soil Health And Climate Data."*

---

## 2. Target Variable Specifications

| Crop | Target Variable | Original Unit | Metric Ton Conversion | Quintal Conversion | Historical Range | Mean ± Std |
|---|---|---|---|---|---|---|
| **Corn (*Zea mays*)** | `Yield` | **Bushels per Acre (bu/acre)** (56 lb/bu) | 1 bu/acre = 0.06277 t/ha (62.77 kg/ha) | 1 bu/acre = 0.2540 quintal/acre | 19.00 – 246.70 bu/acre | 137.38 ± 38.02 bu/acre |
| **Soybean (*Glycine max*)** | `Yield` | **Bushels per Acre (bu/acre)** (60 lb/bu) | 1 bu/acre = 0.06725 t/ha (67.25 kg/ha) | 1 bu/acre = 0.2722 quintal/acre | 13.00 – 82.30 bu/acre | 42.49 ± 10.14 bu/acre |

> [!NOTE]
> In agricultural practice, a bushel is a unit of volume standardized by weight: **56 pounds for shelled corn** and **60 pounds for soybeans**. For international and Indian farming contexts, KrishiYield provides seamless unit toggling to **Quintals per Acre** and **Metric Tons per Hectare (t/ha)**.

---

## 3. Detailed Feature Dictionary

### A. Environmental & Weather Data (Daymet Version 3)
*Source: NASA / Oak Ridge National Laboratory (ORNL) DAAC Daymet Daily Surface Weather Data (1-km grid).*
Converted from daily measurements into 52 weekly averages ($W_{a,b}$, where $a \in [1..6]$ is the weather metric and $b \in [1..52]$ is the calendar week of the year).

| Prefix | Weather Parameter | Raw Unit | Agricultural Significance & Farmer Interpretation |
|---|---|---|---|
| `W11` – `W152` | **Precipitation** | mm / week | Weekly cumulative rainfall and liquid water equivalent. Essential during vegetative growth (weeks 18–26) and critical during flowering/silking (weeks 28–32). Drought during silking causes severe kernel abortion. |
| `W21` – `W252` | **Solar Radiation** | $\text{W/m}^2$ | Solar energy incident on crop canopy. Drives photosynthesis, biomass accumulation, and grain filling. |
| `W31` – `W352` | **Snow Water Equivalent (SWE)** | mm | Water content stored in snowpack. Supplies early spring soil recharge. Note: `W316`–`W345` (summer weeks) are identically 0 in the US Corn Belt and represent zero-dominant non-informative features. |
| `W41` – `W452` | **Maximum Air Temperature** | °C | Daily peak temperature. Heat stress ($T_{\max} > 32^\circ\text{C}$ / $90^\circ\text{F}$) during corn pollination (weeks 28–32) inhibits pollen viability and drastically decreases yield. |
| `W51` – `W552` | **Minimum Air Temperature** | °C | Nighttime temperature. High minimum nighttime temperatures accelerate respiration, burning assimilates needed for grain formation. |
| `W61` – `W652` | **Vapour Pressure** | Pa | Atmospheric moisture deficit indicator. Correlates with evapotranspiration demand and atmospheric drought stress. |

> [!IMPORTANT]
> **Dataset Anomaly Resolved**: In the raw Corn dataset (`corn_data_csv.csv`), column 110 is labeled as `*`. Inspection of surrounding columns (`W32`, `W33`, `*`, `W35`) conclusively proves this is a typographical error for **`W34`** (Snow Water Equivalent for week 4). Our preprocessing pipeline automatically renames this column to `W34`.

---

### B. Soil Health & Profile Attributes (gSSURGO Database)
*Source: USDA Natural Resources Conservation Service (NRCS) Gridded Soil Survey Geographic Database.*
Comprises 10 soil variables measured across 10 depth layers ($S_{a,b}$, where $a \in [1..10]$ is the variable, $b \in [1..10]$ is depth layer):
- Depth Layer 1: 0–5 cm
- Depth Layer 2: 5–10 cm
- Depth Layer 3: 10–20 cm
- Depth Layer 4: 20–30 cm
- Depth Layer 5: 30–40 cm
- Depth Layer 6: 40–60 cm
- Depth Layer 7: 60–80 cm
- Depth Layer 8: 80–100 cm
- Depth Layer 9: 100–150 cm
- Depth Layer 10: 150–200 cm

| Metric | Scientific Parameter | Unit | Farmer-Friendly Definition & Agronomic Impact |
|---|---|---|---|
| `S11` – `S110` | Wet Bulk Density | $\text{g/cm}^3$ | Mass of natural moist soil per volume. High values indicate severe compaction and poor aeration. |
| `S21` – `S210` | Dry Bulk Density | $\text{g/cm}^3$ | Mass of dry soil per volume. Values $> 1.6\,\text{g/cm}^3$ restrict root elongation and water movement. |
| `S31` – `S310` | Clay Percentage | % | Fine mineral fraction ($< 0.002\,\text{mm}$). Increases water and nutrient holding capacity, but excessive clay impedes drainage. |
| `S41` – `S410` | Field Capacity (Upper Limit AWC) | fraction (cm/cm) | Maximum soil water held against gravity. Determines water availability during dry spells. |
| `S51` – `S510` | Permanent Wilting Point (Lower Limit AWC) | fraction (cm/cm) | Soil moisture level below which crop roots can no longer extract water. |
| `S61` – `S610` | Hydraulic Conductivity ($K_{\text{sat}}$) | $\mu\text{m/s}$ | Rate of water infiltration through soil. High conductivity prevents surface ponding; low conductivity causes waterlogging. |
| `S71` – `S710` | Organic Matter Content | % | Humus and decomposed biological matter. Vital for soil structure, nitrogen cycling, and biological health. Ideal topsoil range: 2.5% – 5.0%. |
| `S81` – `S810` | Soil pH | pH scale (0–14) | Acidity/alkalinity of the soil. Optimum nutrient uptake for corn and soybeans occurs at pH **6.0 – 6.8**. Highly acidic soils ($< 5.5$) cause aluminum toxicity; alkaline soils ($> 7.5$) lock phosphorus and micronutrients. |
| `S91` – `S910` | Sand Percentage | % | Coarse mineral particles ($0.05 - 2.0\,\text{mm}$). Provides aeration and permeability, but low nutrient retention. |
| `S101` – `S1010` | Saturated Volumetric Water Content | fraction (cm/cm) | Total soil porosity. Measures total volume of water stored at complete saturation. |

#### Surface Soil Metrics
| Feature | Parameter | Unit | Description |
|---|---|---|---|
| `S_surface1` | Field Slope | % | Field inclination. Steeper slopes cause increased soil erosion, nitrogen runoff, and reduced moisture retention. |
| `S_surface2` | Soil Productivity Index | Index (0–1) | Comprehensive inherent agronomic productivity score based on root zone available water capacity, aeration, and bulk density. |
| `S_surface3` | Average Productivity Index | Index (0–1) | County/regional aggregate soil productivity index. |
| `S_surface4` | Maximum Rooting Depth | cm | Effective depth unobstructed by bedrock, hardpans, or toxic subsoil layers. Corn roots can extend up to 150–180 cm in deep soils. |

---

### C. Farm Management Practices: Planting Progress
*Source: USDA NASS Crop Progress Reports.*
Weekly cumulative percentage of acreage planted within each county:
- **Corn**: `P1` to `P16` (Weeks 14 to 29; early April through mid-July).
- **Soybean**: `P1` to `P14` (Weeks 16 to 29; late April through mid-July).

Agronomic Impact: Early planting allows crops to utilize maximum solar radiation and flower before midsummer heat waves; delayed planting shortens the grain filling window and increases frost risk in autumn.

---

## 4. Feature Engineering & Farmer Translation Strategy

While research models require hundreds of granular sensor inputs, **practical agricultural users think in aggregate seasonal concepts**. KrishiYield bridges this gap through dual interfaces:

1. **Farmer-Intuitive Mode**:
   - Location / Agroecological Region
   - Total Seasonal Rainfall (Low $< 400\,\text{mm}$, Moderate $400-650\,\text{mm}$, High $> 650\,\text{mm}$)
   - Mid-Season Peak Temperature ($< 26^\circ\text{C}$, $26-30^\circ\text{C}$, $> 30^\circ\text{C}$)
   - Topsoil Texture (Silt Loam, Clay Loam, Sandy Loam, Silty Clay)
   - Topsoil Organic Matter % (1.0% to 5.5%)
   - Soil pH (5.5 to 7.8)
   - Planting Date (Early April, Mid April, Early May, Late May)
   - Irrigation Availability (Rainfed vs Supplementary Irrigation)

2. **Agronomist / Scientific Mode**:
   - Full 432/434 parameter inspection, granular weekly weather anomalies, and deep soil profile layers.

Our backend mapping engine translates farmer inputs into mathematically sound, county-calibrated feature distributions, ensuring scientific accuracy without confusing the user.
