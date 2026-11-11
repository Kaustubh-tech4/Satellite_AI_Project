
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


class UNet(nn.Module):
    """Lightweight U-Net for satellite water segmentation."""

    def __init__(self, in_channels=15, base_channels=16):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(
            base_channels * 4, base_channels * 8
        )

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, 2, stride=2
        )
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 2, stride=2
        )
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, 2, stride=2
        )
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # One output logit per pixel: water versus non-water.
        self.output = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output(d1)


if __name__ == "__main__":
    model = UNet(in_channels=15, base_channels=16)
    model.eval()

    with torch.no_grad():
        test_input = torch.zeros(1, 15, 512, 512)
        test_output = model(test_input)

    parameter_count = sum(p.numel() for p in model.parameters())

    print("U-Net model initialized successfully.")
    print("Input shape:", tuple(test_input.shape))
    print("Output shape:", tuple(test_output.shape))
    print(f"Trainable parameters: {parameter_count:,}")
    print("Device: CPU")
