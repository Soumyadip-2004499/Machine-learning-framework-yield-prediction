from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class FullPredictRequest(BaseModel):
    crop: str = Field(..., description="Crop type: 'corn' or 'soybean'")
    features: Dict[str, float] = Field(..., description="Dictionary of feature names and numerical values")

class FarmerPredictRequest(BaseModel):
    crop: str = Field(..., description="Crop type: 'corn' or 'soybean'")
    region: Optional[str] = Field("Central Corn Belt (Iowa/Illinois)", description="Agricultural region or state")
    rainfall_seasonal_mm: float = Field(520.0, ge=100.0, le=1500.0, description="Total expected growing season rainfall in mm")
    temp_peak_c: float = Field(28.0, ge=15.0, le=48.0, description="Average peak daytime temperature during flowering/reproductive stage in °C")
    temp_night_c: float = Field(16.5, ge=5.0, le=32.0, description="Average nighttime minimum temperature in °C")
    solar_radiation: Optional[str] = Field("Normal", description="Solar radiation level: 'Low (Cloudy)', 'Normal', 'High (Abundant Sunlight)'")
    soil_texture: str = Field("Silt Loam", description="Topsoil texture class: 'Silt Loam', 'Clay Loam', 'Loam', 'Sandy Loam', 'Clay'")
    soil_ph: float = Field(6.5, ge=4.5, le=9.0, description="Topsoil pH level (optimal 6.0 - 7.0)")
    organic_matter_pct: float = Field(3.2, ge=0.5, le=8.0, description="Soil organic matter percentage")
    soil_drainage: str = Field("Well-drained", description="Drainage condition: 'Well-drained', 'Moderate', 'Poorly-drained'")
    planting_window: str = Field("Optimal (Late April - Early May)", description="Planting timing: 'Early (Early April)', 'Optimal (Late April - Early May)', 'Late (Late May - June)'")
    irrigation: bool = Field(False, description="Whether supplementary irrigation is available")

class SoilHealthAssessment(BaseModel):
    score: float
    category: str
    organic_matter_status: str
    ph_status: str
    drainage_status: str
    limiting_factors: List[str]

class BenchmarkComparison(BaseModel):
    historical_county_mean: float
    yield_difference: float
    percentage_difference: float
    status: str

class FeatureImpact(BaseModel):
    factor: str
    category: str
    impact: str
    description: str

class PlantingDecision(BaseModel):
    recommendation: str  # "RECOMMENDED", "CAUTION", "HIGH_RISK"
    verdict_badge: str
    verdict_title: str
    breakeven_yield: float
    projected_margin_pct: float
    action_summary: str

class PredictResponse(BaseModel):
    success: bool
    crop: str
    predicted_yield: float
    predicted_yield_kg_ha: float
    predicted_yield_quintal_acre: float
    yield_unit: str
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_name: str
    model_version: str
    benchmark_comparison: BenchmarkComparison
    soil_health: SoilHealthAssessment
    feature_impacts: List[FeatureImpact]
    farmer_guidance: str
    planting_decision: Optional[PlantingDecision] = None
    disclaimer: str
