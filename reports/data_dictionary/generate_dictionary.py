import os
import json
import pandas as pd
import numpy as np

proj_dir = r"C:\Users\chatt\.gemini\antigravity\scratch\krishiyield"
data_dict_dir = os.path.join(proj_dir, "reports", "data_dictionary")
os.makedirs(data_dict_dir, exist_ok=True)

# 1. Weather components mapping
weather_components = {
    1: {"name": "Precipitation", "unit": "mm/week", "desc": "Total weekly precipitation (rain and liquid equivalent)"},
    2: {"name": "Solar Radiation", "unit": "W/m²", "desc": "Weekly mean incident shortwave solar radiation"},
    3: {"name": "Snow Water Equivalent", "unit": "mm", "desc": "Snowpack water equivalent"},
    4: {"name": "Maximum Temperature", "unit": "°C", "desc": "Weekly mean daily maximum 2m air temperature"},
    5: {"name": "Minimum Temperature", "unit": "°C", "desc": "Weekly mean daily minimum 2m air temperature"},
    6: {"name": "Vapour Pressure", "unit": "Pa", "desc": "Weekly mean atmospheric water vapour pressure"}
}

# 2. Soil components mapping (S_a_b where a in 1..10, b in 1..10)
# Depth layers: 1: 0-5cm, 2: 5-10cm, 3: 10-20cm, 4: 20-30cm, 5: 30-40cm, 6: 40-60cm, 7: 60-80cm, 8: 80-100cm, 9: 100-150cm, 10: 150-200cm
depth_layers = {
    1: "0-5 cm", 2: "5-10 cm", 3: "10-20 cm", 4: "20-30 cm", 5: "30-40 cm",
    6: "40-60 cm", 7: "60-80 cm", 8: "80-100 cm", 9: "100-150 cm", 10: "150-200 cm"
}

soil_components = {
    1: {"name": "Wet Soil Bulk Density", "unit": "g/cm³", "desc": "Total mass of moist soil per unit volume"},
    2: {"name": "Dry Bulk Density", "unit": "g/cm³", "desc": "Oven-dry mass of mineral soil per unit volume; reflects soil compaction and porosity"},
    3: {"name": "Clay Content", "unit": "%", "desc": "Percentage of mineral particles smaller than 0.002 mm; controls water retention and cation exchange"},
    4: {"name": "Upper Limit of Plant Available Water Content (Field Capacity)", "unit": "cm/cm (vol fraction)", "desc": "Soil water retention after free drainage ceases"},
    5: {"name": "Lower Limit of Plant Available Water Content (Permanent Wilting Point)", "unit": "cm/cm (vol fraction)", "desc": "Soil water content below which plants can no longer extract water"},
    6: {"name": "Hydraulic Conductivity (Ksat)", "unit": "μm/s", "desc": "Rate of saturated water movement through the soil profile; key for drainage"},
    7: {"name": "Organic Matter Content", "unit": "%", "desc": "Decomposed plant/animal residue; vital for soil fertility, structure, and microbial activity"},
    8: {"name": "Soil pH", "unit": "pH scale (0-14)", "desc": "Negative log of hydrogen ion activity in water; determines nutrient availability (optimal 6.0-7.0)"},
    9: {"name": "Sand Content", "unit": "%", "desc": "Percentage of mineral particles between 0.05 and 2.0 mm; determines aeration and infiltration"},
    10: {"name": "Saturated Volumetric Water Content (Porosity)", "unit": "cm/cm (vol fraction)", "desc": "Total pore space filled with water at saturation"}
}

surface_soil = {
    "S_surface1": {"name": "Field Slope", "unit": "%", "desc": "Average land surface inclination; affects runoff and erosion risk"},
    "S_surface2": {"name": "Soil Productivity Index", "unit": "Index (0-1)", "desc": "Inherent soil quality and yield potential based on root zone properties"},
    "S_surface3": {"name": "Average Productivity Index", "unit": "Index (0-1)", "desc": "County/area-weighted mean productivity index"},
    "S_surface4": {"name": "Rooting Depth", "unit": "cm", "desc": "Maximum effective depth that plant roots can penetrate without physical/chemical restriction"}
}

# 3. Management
# P1 to P16: Planting progress (% of crop planted by that calendar week)
# Corn has P1 to P16 (weeks ~14 to 29 of year, early April to mid July)
# Soybean has P1 to P14 (weeks ~16 to 29 of year, late April to mid July)

data_dict = {
    "project": "KrishiYield: AI-Powered Corn and Soybean Yield Prediction System",
    "sources": {
        "yield_and_management": "USDA National Agricultural Statistics Service (NASS)",
        "weather": "Daymet Version 3: Daily Surface Weather Data on a 1-km Grid for North America (ORNL DAAC)",
        "soil": "Gridded Soil Survey Geographic Database (gSSURGO) - USDA Natural Resources Conservation Service (NRCS)"
    },
    "geography": "105 Corn-and-Soybean Belt counties across 13 US agricultural states (IL, IA, IN, OH, MN, MO, NE, SD, ND, KS, WI, MI, KY)",
    "temporal_range": "1980 to 2018 (39 years)",
    "target_variable": {
        "name": "Yield",
        "units": "bushels per acre (bu/acre)",
        "conversion_factors": {
            "corn_1_bu_acre_to_kg_ha": 62.77,
            "corn_1_bu_acre_to_t_ha": 0.06277,
            "corn_1_bu_acre_to_quintal_acre": 0.2540,
            "soybean_1_bu_acre_to_kg_ha": 67.25,
            "soybean_1_bu_acre_to_t_ha": 0.06725,
            "soybean_1_bu_acre_to_quintal_acre": 0.2722
        }
    },
    "columns": {}
}

# Build column-by-column entries
# Meta
data_dict["columns"]["Loc_Id"] = {"category": "Metadata", "name": "Location Identifier", "unit": "Categorical ID (0-104)", "description": "Anonymized county FIPS index representing agricultural county."}
data_dict["columns"]["Year"] = {"category": "Metadata", "name": "Harvest Year", "unit": "Year (1980-2018)", "description": "Crop production year."}
data_dict["columns"]["Yield"] = {"category": "Target", "name": "Crop Yield", "unit": "bushels per acre (bu/acre)", "description": "Actual harvested crop yield per harvested acre reported by USDA NASS."}

# Weather W11 to W652
for a in range(1, 7):
    comp = weather_components[a]
    for b in range(1, 53):
        col_name = f"W{a}{b}"
        data_dict["columns"][col_name] = {
            "category": "Weather (Daymet)",
            "name": f"{comp['name']} - Week {b}",
            "unit": comp["unit"],
            "description": f"{comp['desc']} during week {b} of the year."
        }
# Note the '*' anomaly in corn
data_dict["columns"]["*"] = {
    "category": "Weather (Daymet) - Anomalous Raw Column",
    "name": "Snow Water Equivalent - Week 4 (misnamed as '*')",
    "unit": "mm",
    "description": "Raw corn CSV typo where column 'W34' was labeled as '*'. Verified by column ordering W32, W33, *, W35."
}

# Soil S11 to S1010
for a in range(1, 11):
    s_comp = soil_components[a]
    for b in range(1, 11):
        col_name = f"S{a}{b}"
        data_dict["columns"][col_name] = {
            "category": "Soil Profile (gSSURGO)",
            "name": f"{s_comp['name']} ({depth_layers[b]})",
            "unit": s_comp["unit"],
            "description": f"{s_comp['desc']} at depth layer {depth_layers[b]}."
        }

for col_name, s_surf in surface_soil.items():
    data_dict["columns"][col_name] = {
        "category": "Surface Soil (gSSURGO)",
        "name": s_surf["name"],
        "unit": s_surf["unit"],
        "description": s_surf["desc"]
    }

# Management P1 to P16
for p in range(1, 17):
    col_name = f"P{p}"
    data_dict["columns"][col_name] = {
        "category": "Crop Management (USDA NASS)",
        "name": f"Planting Progress - Week {p}",
        "unit": "Percentage (%)",
        "description": f"Cumulative percentage of crop planted by planting window week {p}."
    }

# Save JSON
with open(os.path.join(data_dict_dir, "data_dictionary.json"), "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=2)

print("Saved data_dictionary.json with", len(data_dict["columns"]), "feature entries.")
