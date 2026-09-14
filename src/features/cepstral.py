import numpy as np, librosa, scipy.fftpack as fft
from src.utils.config import audio_config

class CepstralFeatureExtractor:
    def __init__(self, n_filters: int = None, n_ceps: int = None, n_fft: int = None, hop_length: int = None):
        self.n_filters = n_filters or audio_config.n_lfcc
        self.n_ceps = n_ceps or audio_config.n_lfcc
        self.n_fft = n_fft or audio_config.n_fft
        self.hop_length = hop_length or audio_config.hop_length

    def extract_lfcc(self, y: np.ndarray, sr: int = 16000) -> np.ndarray:
        frames = librosa.util.frame(y, frame_length=self.n_fft, hop_length=self.hop_length)
        window = np.hanning(self.n_fft)[:, None]
        windowed_frames = frames * window
        spectrum = np.abs(np.fft.rfft(windowed_frames, n=self.n_fft, axis=0))**2
        n_bins = spectrum.shape[0]
        fbank = np.zeros((self.n_filters, n_bins))
        pts = np.linspace(0, n_bins - 1, self.n_filters + 2, dtype=int)
        for i in range(self.n_filters):
            fbank[i, pts[i]:pts[i+1]] = np.linspace(0, 1, pts[i+1] - pts[i])
            fbank[i, pts[i+1]:pts[i+2]] = np.linspace(1, 0, pts[i+2] - pts[i+1])
        energy = np.dot(fbank, spectrum)
        log_energy = np.log(np.maximum(energy, 1e-8))
        static_lfcc = fft.dct(log_energy, type=2, axis=0, norm="ortho")[:self.n_ceps]
        delta_lfcc = librosa.feature.delta(static_lfcc, order=1)
        delta2_lfcc = librosa.feature.delta(static_lfcc, order=2)
        return np.vstack([static_lfcc, delta_lfcc, delta2_lfcc])

    def extract_mfcc(self, y: np.ndarray, sr: int = 16000, n_mfcc: int = 20) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        return np.vstack([mfcc, delta, delta2])
