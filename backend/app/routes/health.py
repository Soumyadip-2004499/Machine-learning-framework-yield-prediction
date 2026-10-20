from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["Health"])

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "CropSense AI Agricultural Crop Prediction Engine",
        "version": "1.0.0"
    }
