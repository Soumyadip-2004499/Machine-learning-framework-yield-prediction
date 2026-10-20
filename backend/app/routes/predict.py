from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from app.schemas.prediction import FullPredictRequest, FarmerPredictRequest, PredictResponse
from app.services.prediction_service import prediction_service

router = APIRouter(prefix="/api", tags=["Prediction"])

@router.post("/predict", response_model=PredictResponse)
async def predict_custom_features(request: FullPredictRequest):
    try:
        # Full feature vector prediction
        crop = request.crop.lower()
        if crop not in ["corn", "soybean"]:
            raise HTTPException(status_code=400, detail=f"Unsupported crop '{request.crop}'. Supported crops are 'corn' and 'soybean'.")
        
        # Format as input payload
        payload = {"crop": crop, **request.features}
        # Run prediction
        res = prediction_service.predict_farmer_simple(payload)
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/farmer-simple", response_model=PredictResponse)
async def predict_farmer_simple(request: FarmerPredictRequest):
    try:
        crop = request.crop.lower()
        if crop not in ["corn", "soybean"]:
            raise HTTPException(status_code=400, detail=f"Unsupported crop '{request.crop}'. Supported crops are 'corn' and 'soybean'.")
        
        payload = request.model_dump()
        res = prediction_service.predict_farmer_simple(payload)
        return res
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/metadata")
async def get_metadata():
    try:
        meta = prediction_service.get_metadata()
        return {"status": "success", "models": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmarks")
async def get_benchmarks():
    return {
        "corn": {
            "historical_mean_bu_acre": 137.38,
            "min_bu_acre": 19.0,
            "max_bu_acre": 246.7,
            "unit": "bushels per acre (bu/acre)",
            "metric_tons_per_ha_mean": 8.62,
            "period": "1980 - 2018 (39 years)",
            "coverage": "105 US Corn Belt Counties"
        },
        "soybean": {
            "historical_mean_bu_acre": 42.49,
            "min_bu_acre": 13.0,
            "max_bu_acre": 82.3,
            "unit": "bushels per acre (bu/acre)",
            "metric_tons_per_ha_mean": 2.86,
            "period": "1980 - 2018 (39 years)",
            "coverage": "105 US Soybean Belt Counties"
        }
    }

@router.get("/dataset/summary")
async def get_dataset_summary(crop: str = "corn"):
    try:
        summary = prediction_service.get_dataset_summary(crop)
        return {"status": "success", "data": summary}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/samples")
async def get_dataset_samples(crop: str = "corn"):
    try:
        samples = prediction_service.get_historical_samples(crop)
        return {"status": "success", "crop": crop, "samples": samples}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/features")
async def get_dataset_features(crop: str = "corn", top_n: int = 10):
    try:
        features = prediction_service.get_top_features(crop, top_n)
        return {"status": "success", "crop": crop, "features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
