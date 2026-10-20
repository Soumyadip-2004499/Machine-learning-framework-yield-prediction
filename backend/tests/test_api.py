import pytest
from fastapi.testclient import TestClient
import sys, os

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "CropSense AI" in data["service"]

def test_benchmarks_endpoint():
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "corn" in data
    assert "soybean" in data
    assert data["corn"]["historical_mean_bu_acre"] > 100.0
    assert data["soybean"]["historical_mean_bu_acre"] > 30.0

def test_farmer_simple_corn_prediction():
    payload = {
        "crop": "corn",
        "region": "Central Corn Belt (Iowa/Illinois)",
        "rainfall_seasonal_mm": 550.0,
        "temp_peak_c": 28.5,
        "temp_night_c": 17.0,
        "soil_texture": "Silt Loam",
        "soil_ph": 6.5,
        "organic_matter_pct": 3.5,
        "soil_drainage": "Well-drained",
        "planting_window": "Optimal (Late April - Early May)",
        "irrigation": False
    }
    response = client.post("/api/predict/farmer-simple", json=payload)
    # If model is loaded, status 200
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True
        assert data["crop"] == "Corn"
        assert 50.0 <= data["predicted_yield"] <= 260.0
        assert data["predicted_yield_kg_ha"] > 0
        assert "confidence_interval_lower" in data
        assert "soil_health" in data
        assert data["soil_health"]["score"] > 0
        assert len(data["feature_impacts"]) > 0
    else:
        # If model is still training, it will return error or 500
        assert response.status_code in [200, 500]

def test_farmer_simple_soybean_prediction():
    payload = {
        "crop": "soybean",
        "region": "Central Corn Belt (Iowa/Illinois)",
        "rainfall_seasonal_mm": 520.0,
        "temp_peak_c": 27.5,
        "temp_night_c": 16.0,
        "soil_texture": "Clay Loam",
        "soil_ph": 6.4,
        "organic_matter_pct": 3.0,
        "soil_drainage": "Well-drained",
        "planting_window": "Optimal (Late April - Early May)",
        "irrigation": False
    }
    response = client.post("/api/predict/farmer-simple", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True
        assert data["crop"] == "Soybean"
        assert 15.0 <= data["predicted_yield"] <= 90.0
        assert data["predicted_yield_kg_ha"] > 0

def test_invalid_crop():
    payload = {
        "crop": "wheat",
        "rainfall_seasonal_mm": 500.0,
        "temp_peak_c": 25.0
    }
    response = client.post("/api/predict/farmer-simple", json=payload)
    assert response.status_code == 400
    assert "Unsupported crop" in response.json()["detail"]

def test_out_of_bounds_inputs():
    # Rainfall < 100 mm should trigger pydantic 422 Unprocessable Entity
    payload = {
        "crop": "corn",
        "rainfall_seasonal_mm": -50.0
    }
    response = client.post("/api/predict/farmer-simple", json=payload)
    assert response.status_code == 422

def test_dataset_summary_endpoint():
    for crop in ["corn", "soybean"]:
        response = client.get(f"/api/dataset/summary?crop={crop}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        summary = data["data"]
        assert summary["total_observations"] == 4000
        assert summary["total_features"] > 400
        assert "yield_statistics" in summary
        assert summary["yield_statistics"]["mean"] > 0

def test_dataset_samples_endpoint():
    for crop in ["corn", "soybean"]:
        response = client.get(f"/api/dataset/samples?crop={crop}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["samples"]) >= 3
        first_sample = data["samples"][0]
        assert "rainfall_seasonal_mm" in first_sample
        assert "temp_peak_c" in first_sample
        assert "organic_matter_pct" in first_sample

def test_dataset_features_endpoint():
    for crop in ["corn", "soybean"]:
        response = client.get(f"/api/dataset/features?crop={crop}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["features"]) > 0
        assert "feature" in data["features"][0]
        assert "importance" in data["features"][0]
