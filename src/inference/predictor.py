import os, time, torch, numpy as np, soundfile as sf, io
from src.models.light_cnn import LightCNN
from src.models.resnet_spec import SEResNetSpectrogram
from src.models.rawnet_mini import RawNetMini
from src.features.preprocessor import SignalPreprocessor
from src.features.spectral import SpectralFeatureExtractor
from src.features.cepstral import CepstralFeatureExtractor
from src.utils.config import path_config

class ForensicPredictor:
    def __init__(self):
        self.preprocessor = SignalPreprocessor()
        self.spectral_extractor = SpectralFeatureExtractor()
        self.cepstral_extractor = CepstralFeatureExtractor()
        self.model_lfcc = LightCNN(in_channels=1, num_classes=2)
        self.model_mel = SEResNetSpectrogram(in_channels=1, num_classes=2)
        self.model_raw = RawNetMini(in_channels=1, num_classes=2)
        self.weights = [0.45, 0.35, 0.20]
        self._load_weights()
        self.model_lfcc.eval()
        self.model_mel.eval()
        self.model_raw.eval()

    def _load_weights(self):
        p_lfcc = os.path.join(path_config.artifacts, "light_cnn_best.pth")
        p_mel = os.path.join(path_config.artifacts, "se_resnet_best.pth")
        p_raw = os.path.join(path_config.artifacts, "rawnet_mini_best.pth")
        if os.path.exists(p_lfcc):
            self.model_lfcc.load_state_dict(torch.load(p_lfcc, map_location="cpu"))
        if os.path.exists(p_mel):
            self.model_mel.load_state_dict(torch.load(p_mel, map_location="cpu"))
        if os.path.exists(p_raw):
            self.model_raw.load_state_dict(torch.load(p_raw, map_location="cpu"))

    def predict_from_audio_array(self, y_raw: np.ndarray, sr: int = 16000, filename: str = "audio_stream") -> dict:
        t0 = time.time()
        y_proc = self.preprocessor.process_pipeline(y_raw, is_training=False)
        mel_mat = self.spectral_extractor.extract_log_mel(y_proc, sr)
        lfcc_mat = self.cepstral_extractor.extract_lfcc(y_proc, sr)
        t_mel = torch.from_numpy(mel_mat).float().unsqueeze(0).unsqueeze(0)
        t_lfcc = torch.from_numpy(lfcc_mat).float().unsqueeze(0).unsqueeze(0)
        t_raw = torch.from_numpy(y_proc).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            prob_lfcc = torch.softmax(self.model_lfcc(t_lfcc), dim=1)[0, 1].item()
            prob_mel = torch.softmax(self.model_mel(t_mel), dim=1)[0, 1].item()
            prob_raw = torch.softmax(self.model_raw(t_raw), dim=1)[0, 1].item()
        w_sum = sum(self.weights)
        spoof_prob = (self.weights[0] * prob_lfcc + self.weights[1] * prob_mel + self.weights[2] * prob_raw) / w_sum
        bonafide_prob = 1.0 - spoof_prob
        prediction = "spoof" if spoof_prob >= 0.5 else "bonafide"
        risk_level = "CRITICAL" if spoof_prob >= 0.85 else "SUSPICIOUS" if spoof_prob >= 0.50 else "AUTHENTIC"
        descriptors = self.spectral_extractor.extract_spectral_descriptors(y_proc, sr)
        latency_ms = (time.time() - t0) * 1000.0
        return {
            "filename": filename,
            "prediction": prediction,
            "confidence_score": round(float(max(spoof_prob, bonafide_prob)), 4),
            "spoof_probability": round(float(spoof_prob), 4),
            "bonafide_probability": round(float(bonafide_prob), 4),
            "risk_level": risk_level,
            "model_breakdown": {
                "light_cnn_lfcc": round(float(prob_lfcc), 4),
                "se_resnet_mel": round(float(prob_mel), 4),
                "rawnet_waveform": round(float(prob_raw), 4)
            },
            "spectral_descriptors": descriptors,
            "inference_latency_ms": round(latency_ms, 2)
        }

    def predict_from_file(self, file_path: str) -> dict:
        y, sr = sf.read(file_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        return self.predict_from_audio_array(y, sr, os.path.basename(file_path))

    def predict_from_bytes(self, audio_bytes: bytes, filename: str = "upload.wav") -> dict:
        buffer = io.BytesIO(audio_bytes)
        y, sr = sf.read(buffer)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        return self.predict_from_audio_array(y, sr, filename)
