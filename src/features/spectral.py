import numpy as np, librosa
from src.utils.config import audio_config

class SpectralFeatureExtractor:
    def __init__(self, n_fft: int = None, hop_length: int = None, n_mels: int = None):
        self.n_fft = n_fft or audio_config.n_fft
        self.hop_length = hop_length or audio_config.hop_length
        self.n_mels = n_mels or audio_config.n_mels

    def extract_stft(self, y: np.ndarray) -> np.ndarray:
        return librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)

    def extract_log_mel(self, y: np.ndarray, sr: int = 16000) -> np.ndarray:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        return librosa.power_to_db(mel, ref=np.max)

    def extract_spectral_descriptors(self, y: np.ndarray, sr: int = 16000) -> dict:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, roll_percent=0.85)[0]
        flat = librosa.feature.spectral_flatness(y=y, n_fft=self.n_fft, hop_length=self.hop_length)[0]
        return {
            "spectral_centroid_mean": float(np.mean(cent)),
            "spectral_centroid_std": float(np.std(cent)),
            "spectral_rolloff_mean": float(np.mean(roll)),
            "spectral_flatness_mean": float(np.mean(flat))
        }
