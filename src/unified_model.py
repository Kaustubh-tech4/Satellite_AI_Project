
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two convolution layers with batch normalization and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UnifiedWaterUNet(nn.Module):
    """
    One water-segmentation model for four standardized input formats.

    Supported modalities:
        rgb     : 3 channels
        optical : 13 channels
        sar     : 2 channels
        fused   : 15 channels (2 SAR + 13 optical)

    Input:
        x: [B, C, H, W]
        modality: one of the strings above

    Output:
        Water logits: [B, 1, H, W]
    """

    def __init__(self, base_channels=16):
        super().__init__()

        self.expected_channels = {
            "rgb": 3,
            "optical": 13,
            "sar": 2,
            "fused": 15,
        }

        # Each sensor type has its own input adapter.
        # All adapters produce the same number of feature channels.
        self.adapters = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(
                    channels, base_channels,
                    kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            )
            for name, channels in self.expected_channels.items()
        })

        c = base_channels
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.enc1 = DoubleConv(c, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.bottleneck = DoubleConv(c * 4, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        self.output = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x, modality):
        if modality not in self.expected_channels:
            raise ValueError(
                f"Unsupported modality: {modality}. "
                f"Choose one of {list(self.expected_channels)}."
            )

        expected = self.expected_channels[modality]

        if x.ndim != 4 or x.shape[1] != expected:
            raise ValueError(
                f"{modality} input must have shape [B, {expected}, H, W]. "
                f"Received {tuple(x.shape)}."
            )

        if x.shape[-2] % 8 != 0 or x.shape[-1] % 8 != 0:
            raise ValueError(
                "Height and width must be divisible by 8."
            )

        x = self.adapters[modality](x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output(d1)


if __name__ == "__main__":
    model = UnifiedWaterUNet(base_channels=16)
    model.eval()

    test_cases = {
        "rgb": 3,
        "optical": 13,
        "sar": 2,
        "fused": 15,
    }

    with torch.no_grad():
        for modality, channels in test_cases.items():
            sample = torch.randn(1, channels, 256, 256)
            prediction = model(sample, modality)

            print(
                f"{modality:8s} | "
                f"input={tuple(sample.shape)} | "
                f"output={tuple(prediction.shape)}"
            )

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {parameters:,}")
