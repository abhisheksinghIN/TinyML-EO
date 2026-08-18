"""
soa.py - model factory for experiments

- Provides a generic PyTorch UNet implemented below (no segmentation-models-pytorch needed for UNet).

"""

# -------------------------
# Quantised-UNet building blocks
# -------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.nn import QuantConv2d, QuantReLU, QuantIdentity

# ---- Quantized DoubleConv ----
class DoubleConvQuant(nn.Module):
    def __init__(self, in_ch, out_ch, w_bits=1, a_bits=1):
        super().__init__()

        self.block = nn.Sequential(
            QuantConv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
                weight_bit_width=w_bits,
            ),
            nn.BatchNorm2d(out_ch),
            QuantReLU(bit_width=a_bits),
            QuantConv2d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
                weight_bit_width=w_bits,
            ),
            nn.BatchNorm2d(out_ch),
            QuantReLU(bit_width=a_bits),
        )

    def forward(self, x):
        return self.block(x)


class DownQuant(nn.Module):
    def __init__(self, in_ch, out_ch, w_bits=1, a_bits=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvQuant(in_ch, out_ch, w_bits, a_bits)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpQuant(nn.Module):
    def __init__(self, in_ch, out_ch, w_bits=1, a_bits=1, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = QuantConv2d(
                in_channels=in_ch // 2,
                out_channels=in_ch // 2,
                kernel_size=2,
                stride=2,
                weight_bit_width=w_bits,
                bias=False,
            )
        self.conv = DoubleConvQuant(in_ch, out_ch, w_bits, a_bits)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConvQuant(nn.Module):
    def __init__(self, in_ch, out_ch, w_bits=8):
        super().__init__()
        self.conv = QuantConv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            bias=True,
            weight_bit_width=w_bits,
        )

    def forward(self, x):
        return self.conv(x)


class UNetQNN(nn.Module):
    def __init__(self, num_classes=6, in_channels=4):
        super().__init__()

        # First block: higher precision (8-bit)
        self.inc = DoubleConvQuant(in_channels, 32, w_bits=8, a_bits=8)

        # Encoder: 1-bit
        self.down1 = DownQuant(32, 64, w_bits=8, a_bits=8)
        self.down2 = DownQuant(64, 128, w_bits=8, a_bits=8)
        self.down3 = DownQuant(128, 256, w_bits=8, a_bits=8)
        self.down4 = DownQuant(256, 512, w_bits=8, a_bits=8)


        # Decoder: 1-bit
        self.up1 = UpQuant(512 + 256, 256, w_bits=8, a_bits=8)
        self.up2 = UpQuant(256 + 128, 128, w_bits=8, a_bits=8)
        self.up3 = UpQuant(128 + 64, 64, w_bits=8, a_bits=8)
        self.up4 = UpQuant(64 + 32, 32, w_bits=8, a_bits=8)

        # Output: 8-bit weights, logits
        self.outc = OutConvQuant(32, num_classes, w_bits=8)

        # Optional: quantized input (if needed by FINN)
        #self.inp_quant = QuantIdentity(bit_width=8, return_quant_tensor=True)

    def forward(self, x):
        #x = self.inp_quant(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# -------------------------
# UNet building blocks
# -------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConvFloat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
class DownFloat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvFloat(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))
class UpFloat(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConvFloat(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConvFloat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
        
        
class UNet(nn.Module):
    def __init__(self, num_classes=6, in_channels=4):
        super().__init__()

        self.inc = DoubleConvFloat(in_channels, 32)
        self.down1 = DownFloat(32, 64)
        self.down2 = DownFloat(64, 128)
        self.down3 = DownFloat(128, 256)
        self.down4 = DownFloat(256, 512)

        self.up1 = UpFloat(512 + 256, 256)
        self.up2 = UpFloat(256 + 128, 128)
        self.up3 = UpFloat(128 + 64, 64)
        self.up4 = UpFloat(64 + 32, 32)

        self.outc = OutConvFloat(32, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

      
                    
# -------------------------
# Model selector (common entry)
# -------------------------
def get_model(model_name, num_classes=4, in_channels=3, img_size=256):
    model_name = model_name.lower()

    if model_name == "unet":
        return UNet(num_classes=num_classes, in_channels=in_channels)
        
    elif model_name == "qnn":
        return UNetQNN(num_classes=num_classes, in_channels=in_channels)


    else:
        raise ValueError(f"Unknown model name: {model_name}")


