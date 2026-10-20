import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJ_DIR, "models")

class PredictionService:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        for crop in ["corn", "soybean"]:
            path = os.path.join(MODEL_DIR, f"{crop}_model_final.pkl")
            if os.path.exists(path):
                print(f"Loading {crop} model from {path}...")
                self.models[crop] = joblib.load(path)
            else:
                print(f"Model file for {crop} not found at {path} yet.")

    def get_metadata(self):
        meta = {}
        for crop, data in self.models.items():
            meta[crop] = {
                "crop": crop,
                "model_name": data["model_name"],
                "model_version": data["model_version"],
                "best_hyperparameters": data["best_params"],
                "metrics": data["metrics"],
                "stats": data["stats"],
                "num_features": data["n_features"]
            }
        return meta

    def assess_soil_health(self, om: float, ph: float, texture: str, drainage: str) -> Dict[str, Any]:
        score = 50.0
        limiting = []

        # Organic Matter assessment (optimal 3.0 - 5.0%)
        if om >= 3.5:
            score += 20
            om_status = "Optimal (>3.5%) - High cation exchange capacity and biological activity"
        elif om >= 2.5:
            score += 15
            om_status = "Adequate (2.5 - 3.5%) - Good nutrient reservoir"
        elif om >= 1.5:
            score += 5
            om_status = "Moderate (1.5 - 2.5%) - Could benefit from cover crops or manure"
            limiting.append("Low Soil Organic Matter reduces water holding capacity")
        else:
            score -= 10
            om_status = "Depleted (<1.5%) - Severe risk of compaction and nutrient leaching"
            limiting.append("Critical Soil Organic Matter deficiency (<1.5%)")

        # pH assessment (optimal 6.2 - 6.8 for corn and soybeans)
        if 6.0 <= ph <= 7.0:
            score += 20
            ph_status = f"Optimal (pH {ph:.1f}) - Maximum availability of N, P, K, and micronutrients"
        elif 5.5 <= ph < 6.0:
            score += 10
            ph_status = f"Slightly Acidic (pH {ph:.1f}) - Tolerable, but phosphorus availability may decrease"
            limiting.append("Mild acidity (pH < 6.0); consider agricultural lime application")
        elif ph < 5.5:
            score -= 15
            ph_status = f"Strongly Acidic (pH {ph:.1f}) - Risk of aluminum toxicity and reduced nodulation"
            limiting.append(f"Severe soil acidity (pH {ph:.1f}) restricts root development")
        elif 7.0 < ph <= 7.8:
            score += 10
            ph_status = f"Slightly Alkaline (pH {ph:.1f}) - Zinc and iron availability may be restricted"
        else:
            score -= 10
            ph_status = f"Alkaline (pH {ph:.1f}) - Micronutrient deficiency risk"
            limiting.append("High soil pH restricts phosphorus and iron availability")

        # Drainage assessment
        if drainage.lower() in ["well-drained", "good"]:
            score += 10
            drainage_status = "Well-drained - Excellent aeration for deep root respiration"
        elif drainage.lower() in ["moderate", "fair"]:
            score += 5
            drainage_status = "Moderately drained - Standard for central prairie soils"
        else:
            score -= 15
            drainage_status = "Poorly-drained - High waterlogging and denitrification risk during heavy spring rains"
            limiting.append("Poor internal drainage risks root anoxia during early season wet spells")

        # Texture adjustment
        if "loam" in texture.lower() or "silt" in texture.lower():
            score = min(100.0, score + 5.0)

        score = max(10.0, min(100.0, score))

        if score >= 85:
            category = "Optimal"
        elif score >= 70:
            category = "Healthy"
        elif score >= 50:
            category = "Moderate"
        else:
            category = "Poor / Degraded"

        return {
            "score": round(score, 1),
            "category": category,
            "organic_matter_status": om_status,
            "ph_status": ph_status,
            "drainage_status": drainage_status,
            "limiting_factors": limiting
        }

    def get_dataset_summary(self, crop: str) -> Dict[str, Any]:
        crop = crop.lower()
        if crop not in ["corn", "soybean"]:
            raise ValueError(f"Unsupported crop '{crop}'.")
        
        if crop == "corn":
            return {
                "crop": "Corn (Zea mays)",
                "total_observations": 4000,
                "total_features": 406,
                "pruned_features": 25,
                "pruned_features_reason": "Zero-variance mid-summer snow water equivalent (W3,21..W3,35)",
                "historical_period": "1980 - 2018 (39 Years)",
                "geographical_coverage": "105 US Corn Belt Counties across 12 States",
                "yield_statistics": {
                    "unit": "bushels per acre (bu/acre)",
                    "mean": 137.38,
                    "std": 38.02,
                    "min": 19.0,
                    "percentile_25": 114.0,
                    "median": 137.0,
                    "percentile_75": 163.0,
                    "max": 246.7
                },
                "weather_aggregates": {
                    "seasonal_precipitation_mean_mm": 524.3,
                    "seasonal_precipitation_range_mm": "210 - 920 mm",
                    "peak_midseason_temp_mean_c": 27.8,
                    "night_minimum_temp_mean_c": 16.4
                },
                "soil_profile_aggregates": {
                    "topsoil_organic_matter_mean_pct": 3.15,
                    "topsoil_ph_mean": 6.48,
                    "predominant_soil_texture": "Silt Loam",
                    "root_zone_depth_cm": "0 - 150 cm"
                },
                "research_paper": "Soumyadip Chattopadhyay, Anurag Sinha, Dr. Iniyan S - Machine Learning Framework For Yield Prediction Using Soil Health And Climate Data"
            }
        else:
            return {
                "crop": "Soybean (Glycine max)",
                "total_observations": 4000,
                "total_features": 409,
                "pruned_features": 21,
                "pruned_features_reason": "Zero-variance mid-summer snow water equivalent (W3,21..W3,35)",
                "historical_period": "1980 - 2018 (39 Years)",
                "geographical_coverage": "105 US Soybean Belt Counties across 13 States",
                "yield_statistics": {
                    "unit": "bushels per acre (bu/acre)",
                    "mean": 42.49,
                    "std": 9.94,
                    "min": 13.0,
                    "percentile_25": 35.8,
                    "median": 42.4,
                    "percentile_75": 49.0,
                    "max": 82.3
                },
                "weather_aggregates": {
                    "seasonal_precipitation_mean_mm": 518.7,
                    "seasonal_precipitation_range_mm": "195 - 905 mm",
                    "peak_midseason_temp_mean_c": 28.1,
                    "night_minimum_temp_mean_c": 16.6
                },
                "soil_profile_aggregates": {
                    "topsoil_organic_matter_mean_pct": 3.22,
                    "topsoil_ph_mean": 6.52,
                    "predominant_soil_texture": "Silt Loam",
                    "root_zone_depth_cm": "0 - 150 cm"
                },
                "research_paper": "S. Iniyan, R. Jebakumar (2021) - Mutual Information Feature Selection (MIFS) Based Crop Yield Prediction on Corn and Soybean Crops Using Multilayer Stacked Ensemble Regression (MSER)"
            }

    def get_historical_samples(self, crop: str) -> List[Dict[str, Any]]:
        crop = crop.lower()
        if crop == "corn":
            return [
                {
                    "id": "corn_bountiful_2017",
                    "title": "2017 Record Corn Belt Harvest",
                    "year": 2017,
                    "location": "Iowa / Central Illinois",
                    "actual_yield_bu_acre": 204.1,
                    "actual_yield_kg_ha": 12811.4,
                    "rainfall_seasonal_mm": 560,
                    "temp_peak_c": 28.2,
                    "temp_night_c": 16.8,
                    "organic_matter_pct": 4.2,
                    "soil_ph": 6.6,
                    "soil_texture": "Silt Loam",
                    "soil_drainage": "Well-drained",
                    "planting_window": "Optimal (Late April - Early May)",
                    "irrigation": False,
                    "scenario_tag": "bountiful",
                    "notes": "Optimal rainfall distribution and mild July pollination temperatures produced an outstanding grain fill."
                },
                {
                    "id": "corn_drought_2012",
                    "title": "2012 Severe Midwest Drought",
                    "year": 2012,
                    "location": "Central Corn Belt",
                    "actual_yield_bu_acre": 104.5,
                    "actual_yield_kg_ha": 6559.5,
                    "rainfall_seasonal_mm": 270,
                    "temp_peak_c": 35.1,
                    "temp_night_c": 20.4,
                    "organic_matter_pct": 3.1,
                    "soil_ph": 6.5,
                    "soil_texture": "Silt Loam",
                    "soil_drainage": "Well-drained",
                    "planting_window": "Optimal (Late April - Early May)",
                    "irrigation": False,
                    "scenario_tag": "drought",
                    "notes": "Historic heat wave (>35°C) during silking caused severe kernel desiccation and reduced ear count."
                },
                {
                    "id": "corn_average_2004",
                    "title": "2004 Baseline Climatological Season",
                    "year": 2004,
                    "location": "Indiana / Ohio Plain",
                    "actual_yield_bu_acre": 160.0,
                    "actual_yield_kg_ha": 10043.2,
                    "rainfall_seasonal_mm": 510,
                    "temp_peak_c": 27.5,
                    "temp_night_c": 16.2,
                    "organic_matter_pct": 3.4,
                    "soil_ph": 6.5,
                    "soil_texture": "Silt Loam",
                    "soil_drainage": "Well-drained",
                    "planting_window": "Optimal (Late April - Early May)",
                    "irrigation": False,
                    "scenario_tag": "average",
                    "notes": "Near-perfect 39-year historical baseline conditions across the Corn Belt."
                },
                {
                    "id": "corn_flood_1993",
                    "title": "1993 Great Midwest Flood Season",
                    "year": 1993,
                    "location": "Upper Mississippi Basin",
                    "actual_yield_bu_acre": 112.0,
                    "actual_yield_kg_ha": 7030.2,
                    "rainfall_seasonal_mm": 880,
                    "temp_peak_c": 25.5,
                    "temp_night_c": 15.0,
                    "organic_matter_pct": 3.5,
                    "soil_ph": 6.4,
                    "soil_texture": "Clay Loam",
                    "soil_drainage": "Moderate",
                    "planting_window": "Late (Late May - June)",
                    "irrigation": False,
                    "scenario_tag": "flood",
                    "notes": "Persistent saturation caused root hypoxia, denitrification, and delayed maturation."
                }
            ]
        else:
            return [
                {
                    "id": "soy_bountiful_2016",
                    "title": "2016 Record Soybean Pod Season",
                    "year": 2016,
                    "location": "Illinois / Iowa",
                    "actual_yield_bu_acre": 58.4,
                    "actual_yield_kg_ha": 3927.4,
                    "rainfall_seasonal_mm": 550,
                    "temp_peak_c": 27.8,
                    "temp_night_c": 17.0,
                    "organic_matter_pct": 4.1,
                    "soil_ph": 6.7,
                    "soil_texture": "Silt Loam",
                    "soil_drainage": "Well-drained",
                    "planting_window": "Optimal (Late April - Early May)",
                    "irrigation": False,
                    "scenario_tag": "bountiful",
                    "notes": "August rainfall was exceptional during pod-fill (R4-R6 stages), maximizing 100-seed weights."
                },
                {
                    "id": "soy_drought_2012",
                    "title": "2012 Severe Midwest Drought",
                    "year": 2012,
                    "location": "Western Soybean Belt",
                    "actual_yield_bu_acre": 28.2,
                    "actual_yield_kg_ha": 1896.5,
                    "rainfall_seasonal_mm": 260,
                    "temp_peak_c": 34.8,
                    "temp_night_c": 19.8,
                    "organic_matter_pct": 3.0,
                    "soil_ph": 6.4,
                    "soil_texture": "Silt Loam",
                    "soil_drainage": "Well-drained",
                    "planting_window": "Optimal (Late April - Early May)",
                    "irrigation": False,
                    "scenario_tag": "drought",
                    "notes": "Excessive evapotranspiration in August caused severe flower abortion and pod shattering."
                },
                {
                    "id": "soy_average_2005",
                    "title": "2005 Baseline Climatological Season",
                    "year": 2005,
                    "location": "Eastern Soybean Belt",
                    "actual_yield_bu_acre": 43.1,
                    "actual_yield_kg_ha": 2898.5,
                    "rainfall_seasonal_mm": 490,
                    "temp_peak_c": 28.0,
                    "temp_night_c": 16.5,
                    "organic_matter_pct": 3.3,
                    "soil_ph": 6.5,
                    "soil_texture": "Silt Loam",
                    "soil_drainage": "Well-drained",
                    "planting_window": "Optimal (Late April - Early May)",
                    "irrigation": False,
                    "scenario_tag": "average",
                    "notes": "Representative 39-year baseline season with steady vegetative development."
                }
            ]

    def get_top_features(self, crop: str, top_n: int = 10) -> List[Dict[str, Any]]:
        crop = crop.lower()
        if crop not in self.models:
            return []
        
        artifact = self.models[crop]
        model = artifact["model"]
        features = artifact["features"]
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1][:top_n]
        
        desc_lookup = {
            "W428": "Week 28 Peak Air Temp (°C) — Critical Silking / Flowering Stage",
            "W429": "Week 29 Peak Air Temp (°C) — Pollen Shedding & Fertilization",
            "W430": "Week 30 Peak Air Temp (°C) — Early Kernel Blister Stage",
            "W431": "Week 31 Peak Air Temp (°C) — Kernel Milk Stage Sensitivity",
            "W432": "Week 32 Peak Air Temp (°C) — Mid-Season Dough Stage",
            "W433": "Week 33 Peak Air Temp (°C) — Soybean Pod Fill Stage",
            "W620": "Week 20 Vapor Pressure Deficit / Radiation — Vegetative Expansion",
            "W228": "Week 28 Weekly Rainfall (mm) — Reproductive Moisture Supply",
            "W232": "Week 32 Weekly Rainfall (mm) — Grain Filling Moisture",
            "S_surface2": "Surface Available Water Storage (AWS) — Upper Root Buffer",
            "S71": "Topsoil Organic Matter Layer 1 (0–5 cm) — Nutrient Reservoir",
            "S81": "Topsoil pH Layer 1 (0–5 cm) — Macronutrient Bioavailability",
            "P6": "Phenology P6 — Physiological Maturity Calendar Day",
            "P9": "Phenology P9 — USDA Crop Progress Milestone",
            "P10": "Phenology P10 — Crop Development Rate Index",
            "P12": "Phenology P12 — Harvest Timing & Dry-Down Duration"
        }
        
        results = []
        for idx in sorted_indices:
            feat_name = features[idx]
            imp_val = float(importances[idx])
            desc = desc_lookup.get(feat_name, f"Agroclimatic Feature {feat_name}")
            category = "Weather (Temperature)" if feat_name.startswith("W4") else (
                "Weather (Precipitation)" if feat_name.startswith("W2") else (
                "Weather (VPD/Radiation)" if feat_name.startswith("W6") else (
                "Soil Horizon" if feat_name.startswith("S") else "Crop Phenology"
            )))
            results.append({
                "feature": feat_name,
                "importance": round(imp_val, 4),
                "percentage": round(imp_val * 100.0, 2),
                "category": category,
                "description": desc
            })
        return results

    def predict_farmer_simple(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        crop = payload.get("crop", "corn").lower()
        if crop not in self.models:
            self.load_models()
        if crop not in self.models:
            raise ValueError(f"Model for {crop} not loaded or unavailable.")

        artifact = self.models[crop]
        model = artifact["model"]
        feature_cols = artifact["features"]
        feature_means = artifact["feature_means"]
        feature_stds = artifact["feature_stds"]
        metrics = artifact["metrics"]
        stats = artifact["stats"]

        # Parse farmer simple inputs
        rainfall_mm = float(payload.get("rainfall_seasonal_mm", 520.0))
        temp_peak = float(payload.get("temp_peak_c", 28.0))
        temp_night = float(payload.get("temp_night_c", 16.5))
        om_pct = float(payload.get("organic_matter_pct", 3.2))
        soil_ph = float(payload.get("soil_ph", 6.5))
        texture = payload.get("soil_texture", "Silt Loam")
        drainage = payload.get("soil_drainage", "Well-drained")
        planting = payload.get("planting_window", "Optimal (Late April - Early May)")
        irrigation = bool(payload.get("irrigation", False))

        # Build feature vector initialized with training means
        x_vec = np.array([feature_means.get(col, 0.0) for col in feature_cols], dtype=np.float32)

        # Effective rainfall and microclimate cooling accounting for irrigation
        # In agronomic reality, irrigation supplies ~150 mm water during critical grain fill
        # and transpirational cooling reduces canopy temperature by ~2.2°C, protecting silking/pollination
        effective_rain = rainfall_mm + (150.0 if irrigation else 0.0)
        effective_tpeak = temp_peak - (2.2 if irrigation else 0.0)

        rain_scale = effective_rain / 520.0

        # 1. Map Precipitation features (W1)
        # As documented in Iniyan et al. (2021) Section 3:
        # Wab: a=1 is Precipitation, b=1..52 is week of year.
        # Growing season spans weeks 18 to 38 (May through September).
        for i, col in enumerate(feature_cols):
            if col.startswith("W1"):
                try:
                    w = int(col[2:])
                    if 18 <= w <= 38:
                        x_vec[i] *= rain_scale
                    else:
                        # Off-season precipitation moderately affects subsoil recharge
                        x_vec[i] *= (1.0 + (rain_scale - 1.0) * 0.3)
                except ValueError:
                    x_vec[i] *= rain_scale

        # 2. Map Maximum Daytime Temperature (W4)
        # a=4 is Maximum Temperature (°C).
        # Shift reproductive peak weeks (26 to 34: July flowering/silking to August pod-fill/dough stage)
        temp_peak_shift = effective_tpeak - 28.0
        for i, col in enumerate(feature_cols):
            if col.startswith("W4"):
                try:
                    week_num = int(col[2:])
                    if 26 <= week_num <= 34:
                        x_vec[i] += temp_peak_shift
                    elif 18 <= week_num <= 38:
                        x_vec[i] += (temp_peak_shift * 0.6)
                    else:
                        x_vec[i] += (temp_peak_shift * 0.25)
                except ValueError:
                    x_vec[i] += temp_peak_shift

        # 3. Map Night Minimum Temperature (W5)
        # a=5 is Minimum Temperature (°C). High night temps increase dark respiration loss.
        temp_night_shift = temp_night - 16.5
        for i, col in enumerate(feature_cols):
            if col.startswith("W5"):
                try:
                    week_num = int(col[2:])
                    if 18 <= week_num <= 38:
                        x_vec[i] += temp_night_shift
                except ValueError:
                    x_vec[i] += temp_night_shift

        # 4. Map Root-Zone Available Water Storage (AWS: S_surface2 and S4)
        # In dry seasons, soil available water capacity is depleted.
        aws_factor = min(1.25, max(0.35, rain_scale))
        for i, col in enumerate(feature_cols):
            if col == "S_surface2" or col.startswith("S4"):
                x_vec[i] *= aws_factor

        # 5. Map Soil Organic Matter (S71..S710, S_surface1)
        om_ratio = om_pct / 3.2
        for i, col in enumerate(feature_cols):
            if col.startswith("S7") or col == "S_surface1":
                x_vec[i] *= om_ratio

        # 6. Map Soil pH (S81..S810)
        ph_shift = soil_ph - 6.5
        for i, col in enumerate(feature_cols):
            if col.startswith("S8"):
                x_vec[i] += ph_shift

        # 7. Map Texture adjustments
        if "sandy" in texture.lower():
            # Sandy: higher sand (S9), lower water capacity (S4, S_surface2)
            for i, col in enumerate(feature_cols):
                if col.startswith("S9"):
                    x_vec[i] *= 1.4
                elif col.startswith("S4") or col == "S_surface2":
                    x_vec[i] *= 0.70
        elif "clay" in texture.lower():
            # Heavy clay: higher clay (S3), lower Ksat (S6)
            for i, col in enumerate(feature_cols):
                if col.startswith("S3"):
                    x_vec[i] *= 1.35
                elif col.startswith("S6"):
                    x_vec[i] *= 0.60

        # 8. Map Planting Progress (P1..P16)
        # Shift weekly planting percentages based on early vs late window
        planting_lower = planting.lower()
        is_optimal = "optimal" in planting_lower
        is_early = not is_optimal and "early" in planting_lower
        is_late = not is_optimal and ("late" in planting_lower or "june" in planting_lower)

        if is_early:
            for i, col in enumerate(feature_cols):
                if col in ["P3", "P4"]:
                    x_vec[i] += 5.0
                elif col in ["P8", "P9"]:
                    x_vec[i] = max(0.0, x_vec[i] - 5.0)
        elif is_late:
            for i, col in enumerate(feature_cols):
                if col in ["P9", "P10", "P11"]:
                    x_vec[i] += 6.0
                elif col in ["P5", "P6", "P7"]:
                    x_vec[i] = max(0.0, x_vec[i] - 6.0)

        # Reshape and predict
        x_input = x_vec.reshape(1, -1)
        pred_yield = float(model.predict(x_input)[0])

        # Agronomic adjustment for late planting truncated solar radiation
        if is_late:
            pred_yield *= 0.90
        elif is_early:
            pred_yield *= 0.98

        # Enforce physical biological bounds
        if crop == "corn":
            pred_yield = max(20.0, min(250.0, pred_yield))
        else:
            pred_yield = max(10.0, min(90.0, pred_yield))

        # Conversions
        # Corn: 1 bu = 56 lbs = 25.40 kg = 0.2540 quintal; 1 bu/acre = 62.77 kg/ha
        # Soybean: 1 bu = 60 lbs = 27.22 kg = 0.2722 quintal; 1 bu/acre = 67.25 kg/ha
        if crop == "corn":
            pred_kg_ha = pred_yield * 62.77
            pred_quintal = pred_yield * 0.2540
        else:
            pred_kg_ha = pred_yield * 67.25
            pred_quintal = pred_yield * 0.2722

        # Confidence interval (95% prediction interval: +/- 1.96 * test_rmse)
        rmse = metrics["test_rmse"]
        lower_bound = max(0.0, pred_yield - 1.96 * rmse)
        upper_bound = pred_yield + 1.96 * rmse

        # Benchmark comparison
        hist_mean = stats["yield_mean"]
        diff = pred_yield - hist_mean
        diff_pct = (diff / hist_mean) * 100.0
        if diff_pct > 5.0:
            bench_status = "Above Historical County Average"
        elif diff_pct < -5.0:
            bench_status = "Below Historical County Average"
        else:
            bench_status = "Near Historical County Average"

        # Soil health
        soil_assessment = self.assess_soil_health(om_pct, soil_ph, texture, drainage)

        # Feature impacts
        impacts = []
        if temp_peak > 31.0:
            impacts.append({
                "factor": "Mid-Season Peak Temperature",
                "category": "Weather",
                "impact": "Negative",
                "description": f"Daytime temperatures of {temp_peak:.1f}°C during flowering/silking induce pollen desiccation and kernel abortion."
            })
        elif 24.0 <= temp_peak <= 29.0:
            impacts.append({
                "factor": "Optimal Daytime Temperature",
                "category": "Weather",
                "impact": "Positive",
                "description": f"Daytime temperature of {temp_peak:.1f}°C aligns with the physiological optimum for photosynthetic biomass conversion."
            })

        if rainfall_mm < 380.0 and not irrigation:
            impacts.append({
                "factor": "Seasonal Moisture Deficit",
                "category": "Water",
                "impact": "Negative",
                "description": f"Seasonal precipitation of {rainfall_mm:.0f} mm is insufficient for peak yield without irrigation."
            })
        elif rainfall_mm >= 450.0:
            impacts.append({
                "factor": "Adequate Moisture Supply",
                "category": "Water",
                "impact": "Positive",
                "description": f"Plentiful rainfall ({rainfall_mm:.0f} mm) sustains grain filling and avoids stomatal closure."
            })

        if om_pct >= 3.0:
            impacts.append({
                "factor": "Topsoil Organic Matter",
                "category": "Soil",
                "impact": "Positive",
                "description": f"Organic matter at {om_pct:.1f}% enhances moisture retention during dry spells."
            })

        # Plain language guidance
        guidance_parts = []
        if temp_peak > 31.0:
            guidance_parts.append("Heat stress mitigation: Consider shifting hybrid selection towards heat-tolerant maturities or planting earlier to pollinate before late July peak heat.")
        if rainfall_mm < 400.0 and not irrigation:
            guidance_parts.append("Water conservation: Prioritize conservation tillage or residue cover to minimize evaporation.")
        if soil_ph < 6.0:
            guidance_parts.append(f"Soil adjustment: Soil pH of {soil_ph:.1f} is acidic. Applying agricultural limestone can lift nutrient availability.")
        if not guidance_parts:
            guidance_parts.append("Favorable agronomic conditions: Current climate and soil parameters indicate strong yield potential. Maintain standard nutrient and weed management.")

        # Farmer Planting Decision & Break-Even Economics
        breakeven = 115.0 if crop == "corn" else 34.0
        margin_pct = ((pred_yield - breakeven) / breakeven) * 100.0

        if pred_yield < (breakeven * 0.98) or (rainfall_mm < 280.0 and not irrigation and temp_peak >= 33.0):
            decision_rec = "HIGH_RISK"
            decision_badge = "HIGH RISK / NOT ADVISED"
            decision_title = f"High Risk of Crop Loss — Planting Not Advised"
            decision_action = f"Projected yield of {pred_yield:.1f} bu/ac falls below the estimated {breakeven:.1f} bu/ac operational break-even threshold due to severe climatic/soil stress. Planting without irrigation or soil remediation poses high financial loss risk."
        elif pred_yield < hist_mean or soil_assessment["score"] < 65:
            decision_rec = "CAUTION"
            decision_badge = "PROCEED WITH CAUTION"
            decision_title = f"Viable to Plant — Mitigation Recommended"
            decision_action = f"Projected harvest of {pred_yield:.1f} bu/ac is economically viable ({margin_pct:+.1f}% above {breakeven:.1f} bu/ac break-even), but sits below the 105-county historical average of {hist_mean:.1f} bu/ac. Follow soil fertility and water conservation steps to raise yield."
        else:
            decision_rec = "RECOMMENDED"
            decision_badge = "HIGHLY RECOMMENDED"
            decision_title = f"Favorable Conditions — Proceed with Planting"
            decision_action = f"Projected harvest of {pred_yield:.1f} bu/ac surpasses the 105-county historical benchmark ({hist_mean:.1f} bu/ac) with a strong +{margin_pct:.1f}% profit margin above operating break-even. Optimal conditions for a bountiful harvest."

        decision_obj = {
            "recommendation": decision_rec,
            "verdict_badge": decision_badge,
            "verdict_title": decision_title,
            "breakeven_yield": breakeven,
            "projected_margin_pct": round(margin_pct, 1),
            "action_summary": decision_action
        }

        return {
            "success": True,
            "crop": crop.capitalize(),
            "predicted_yield": round(pred_yield, 2),
            "predicted_yield_kg_ha": round(pred_kg_ha, 1),
            "predicted_yield_quintal_acre": round(pred_quintal, 2),
            "yield_unit": "bushels per acre (bu/acre)",
            "confidence_interval_lower": round(lower_bound, 2),
            "confidence_interval_upper": round(upper_bound, 2),
            "model_name": artifact["model_name"],
            "model_version": artifact["model_version"],
            "benchmark_comparison": {
                "historical_county_mean": round(hist_mean, 2),
                "yield_difference": round(diff, 2),
                "percentage_difference": round(diff_pct, 1),
                "status": bench_status
            },
            "soil_health": soil_assessment,
            "feature_impacts": impacts,
            "farmer_guidance": " ".join(guidance_parts),
            "planting_decision": decision_obj,
            "disclaimer": "Predictions are data-driven estimates derived from historical USDA-NASS, Daymet, and gSSURGO models. Actual yields may vary due to extreme local weather, pest pressures, and field management practices."
        }

# Global singleton
prediction_service = PredictionService()
