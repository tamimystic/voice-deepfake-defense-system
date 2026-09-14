import io
from fastapi import APIRouter, File, UploadFile, HTTPException
from src.api.schemas import HealthResponse, DetectionResponse, ForensicsResponse
from src.inference.predictor import ForensicPredictor

router = APIRouter(prefix="/api/v1")
predictor = ForensicPredictor()

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse()

@router.post("/detect", response_model=DetectionResponse)
async def detect_deepfake(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".flac", ".wav", ".mp3", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported audio format. Supported: .flac, .wav, .mp3, .ogg")
    content = await file.read()
    result = predictor.predict_from_bytes(content, filename=file.filename)
    return result

@router.post("/forensics", response_model=ForensicsResponse)
async def deep_forensics(file: UploadFile = File(...)):
    content = await file.read()
    det = predictor.predict_from_bytes(content, filename=file.filename)
    summary = {
        "vocoder_artifact_alert": det["spectral_descriptors"]["spectral_flatness_mean"] > 0.05,
        "high_frequency_suppression": det["spectral_descriptors"]["spectral_rolloff_mean"] < 4500,
        "recommendation": "BLOCK_VOICE_AUTH" if det["prediction"] == "spoof" else "PASS_VERIFICATION"
    }
    return {"detection": det, "forensic_summary": summary}
