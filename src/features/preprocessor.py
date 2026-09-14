import numpy as np, librosa
from src.utils.config import audio_config

class SignalPreprocessor:
    def __init__(self, target_samples: int = None, alpha: float = None, vad_top_db: int = None):
        self.target_samples = target_samples or audio_config.target_samples
        self.alpha = alpha or audio_config.pre_emphasis_alpha
        self.vad_top_db = vad_top_db or audio_config.vad_top_db

    def pre_emphasis(self, y: np.ndarray) -> np.ndarray:
        if len(y) == 0:
            return y
        return np.append(y[0], y[1:] - self.alpha * y[:-1])

    def trim_silence_vad(self, y: np.ndarray) -> np.ndarray:
        intervals = librosa.effects.split(y=y, top_db=self.vad_top_db)
        if len(intervals) == 0:
            return y
        trimmed = np.concatenate([y[start:end] for start, end in intervals])
        return trimmed if len(trimmed) > 1000 else y

    def fix_length(self, y: np.ndarray, is_training: bool = False) -> np.ndarray:
        n = len(y)
        if n == self.target_samples:
            return y
        if n > self.target_samples:
            if is_training:
                start = np.random.randint(0, n - self.target_samples + 1)
            else:
                start = (n - self.target_samples) // 2
            return y[start:start + self.target_samples]
        pad_width = self.target_samples - n
        return np.pad(y, (0, pad_width), mode="wrap")

    def peak_normalize(self, y: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(y)) + 1e-7
        return y / peak

    def process_pipeline(self, y: np.ndarray, is_training: bool = False) -> np.ndarray:
        y_filt = self.pre_emphasis(y)
        y_trim = self.trim_silence_vad(y_filt)
        y_fixed = self.fix_length(y_trim, is_training=is_training)
        return self.peak_normalize(y_fixed)
