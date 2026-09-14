import os
from dataclasses import dataclass, field

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    target_duration: float = 4.0
    target_samples: int = 64000
    pre_emphasis_alpha: float = 0.97
    vad_top_db: int = 40
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    n_lfcc: int = 20
    n_cqt_bins: int = 84

@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    epochs: int = 15
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    num_cpu_threads: int = 6
    device: str = "cpu"
    seed: int = 42

@dataclass
class PathConfig:
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_raw: str = field(init=False)
    data_processed: str = field(init=False)
    artifacts: str = field(init=False)
    checkpoints: str = field(init=False)

    def __post_init__(self):
        self.data_raw = os.path.join(self.base_dir, "data", "raw")
        self.data_processed = os.path.join(self.base_dir, "data", "processed")
        self.artifacts = os.path.join(self.base_dir, "artifacts")
        self.checkpoints = os.path.join(self.base_dir, "checkpoints")
        os.makedirs(self.data_processed, exist_ok=True)
        os.makedirs(self.artifacts, exist_ok=True)
        os.makedirs(self.checkpoints, exist_ok=True)

audio_config = AudioConfig()
train_config = TrainingConfig()
path_config = PathConfig()
