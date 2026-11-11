
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two convolution layers with batch normalization and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class RGBUNet(nn.Module):
    """
    U-Net for RGB satellite flood segmentation.

    Input:
        (batch_size, 3, height, width)

    Output:
        (batch_size, 1, height, width)

    Output values are raw logits.
    Apply sigmoid to obtain water probabilities.
    """

    def __init__(self, in_channels=3, base_channels=8):
        super().__init__()

        if in_channels != 3:
            raise ValueError(
                "RGBUNet requires exactly 3 input channels."
            )

        if base_channels < 1:
            raise ValueError(
                "base_channels must be at least 1."
            )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Encoder
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = DoubleConv(c4, c4 * 2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            c4 * 2, c4, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(c4 * 2, c4)

        self.up3 = nn.ConvTranspose2d(
            c4, c3, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(c3 * 2, c3)

        self.up2 = nn.ConvTranspose2d(
            c3, c2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(c2 * 2, c2)

        self.up1 = nn.ConvTranspose2d(
            c2, c1, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(c1 * 2, c1)

        # One output channel: water segmentation logits
        self.output = nn.Conv2d(
            c1, 1, kernel_size=1
        )

    def forward(self, x):

        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "Expected input shape (N, 3, H, W), "
                f"received {tuple(x.shape)}"
            )

        if x.shape[-2] % 16 != 0 or x.shape[-1] % 16 != 0:
            raise ValueError(
                "Height and width must be divisible by 16."
            )

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


# --------------------------------------------------
# MODEL TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 50)
    print("RGB U-NET MODEL TEST")
    print("=" * 50)

    # Use CPU for your current environment.
    device = torch.device("cpu")

    model = RGBUNet(
        in_channels=3,
        base_channels=8,
    ).to(device)

    model.eval()

    # Simulate one 512 x 512 RGB satellite image.
    dummy_input = torch.rand(
        1, 3, 512, 512, device=device
    )

    with torch.no_grad():
        logits = model(dummy_input)
        probabilities = torch.sigmoid(logits)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Device: {device}")
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(
        "Probability shape: "
        f"{tuple(probabilities.shape)}"
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(
        f"Probability range: "
        f"{probabilities.min().item():.4f} - "
        f"{probabilities.max().item():.4f}"
    )

    assert logits.shape == (1, 1, 512, 512)

    print("\nRGB U-Net model test successful.")
    print("No model training was performed.")
