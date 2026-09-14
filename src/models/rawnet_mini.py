import torch, torch.nn as nn, numpy as np

class SincConv1D(nn.Module):
    def __init__(self, out_channels: int = 64, kernel_size: int = 129, sample_rate: int = 16000, min_low_hz: int = 50, min_band_hz: int = 50):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / self.kernel_size)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * np.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return nn.functional.conv1d(waveforms, filters, stride=1, padding=self.kernel_size // 2)

class RawNetMini(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.sinc_conv = SincConv1D(out_channels=64, kernel_size=129, sample_rate=16000)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.lrelu(self.bn1(self.pool1(torch.abs(self.sinc_conv(x)))))
        x = self.pool2(self.lrelu(self.bn2(self.conv1(x))))
        x = self.pool3(self.lrelu(self.bn3(self.conv2(x))))
        x = self.global_pool(x).flatten(1)
        return self.fc(x)
