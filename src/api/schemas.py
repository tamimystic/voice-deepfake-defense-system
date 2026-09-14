from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "healthy"
    device: str = "cpu"
    threads_allocated: int = 6
    version: str = "1.0.0"
    active_models: list[str] = ["light_cnn_lfcc", "se_resnet_mel", "rawnet_mini"]

class ModelBreakdown(BaseModel):
    light_cnn_lfcc: float
    se_resnet_mel: float
    rawnet_waveform: float

class SpectralDescriptors(BaseModel):
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_rolloff_mean: float
    spectral_flatness_mean: float

class DetectionResponse(BaseModel):
    filename: str
    prediction: str
    confidence_score: float
    spoof_probability: float
    bonafide_probability: float
    risk_level: str
    model_breakdown: ModelBreakdown
    spectral_descriptors: SpectralDescriptors
    inference_latency_ms: float

class ForensicsResponse(BaseModel):
    detection: DetectionResponse
    forensic_summary: dict
