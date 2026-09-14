import torch, torch.nn as nn, torch.nn.functional as F

class MaxFeatureMap2D(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(1)
        out1, out2 = torch.split(x, c // 2, dim=1)
        return torch.max(out1, out2)

class LightCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.mfm1 = MaxFeatureMap2D()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.mfm2 = MaxFeatureMap2D()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.mfm3 = MaxFeatureMap2D()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.mfm4 = MaxFeatureMap2D()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.mfm1(self.conv1(x)))
        x = self.pool2(self.mfm2(self.conv2(x)))
        x = self.pool3(self.mfm3(self.conv3(x)))
        x = self.mfm4(self.conv4(x))
        x = self.global_pool(x).flatten(1)
        return self.fc(x)
