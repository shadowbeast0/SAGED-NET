import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class HINBlock(nn.Module):
    """
    Half-Instance Normalization Block with optional Dropout and Squeeze-Excitation.
    """
    def __init__(self, channels: int, drop_rate: float = 0.0, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.inorm = nn.InstanceNorm2d(channels // 2, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
        self.drop = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()
        self.se = SEModule(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split channels for Half-Instance Norm
        c = x.size(1) // 2
        a, b = x[:, :c], x[:, c:]
        a = self.inorm(a)
        x_norm = torch.cat([a, b], dim=1)

        y = self.act(self.bn1(self.conv1(x_norm)))
        y = self.drop(self.act(self.bn2(self.conv2(y))))
        y = self.se(y)
        return x_norm + y


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, drop_rate: float = 0.0):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.hin1 = HINBlock(out_channels, drop_rate)
        self.hin2 = HINBlock(out_channels, drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.downsample(x)))
        x = self.hin1(x)
        x = self.hin2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        
        self.deconv = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 
            kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.se = SEModule(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.deconv(x)))
        x = self.act(self.bn3(self.conv2(x)))
        x = self.se(x)
        return x


class SAGEDNet(nn.Module):
    """
    Structural Adaptive Gated Encoder-Decoder Network (SAGED-Net).
    Features learnable gating for adaptive feature fusion between stages.
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        base_channels: int = 32, 
        stages: int = 4, 
        num_classes: int = 1, 
        drop_rate: float = 0.1, 
        learnable_gating: bool = True
    ):
        super().__init__()
        self.stages = stages
        
        # Channel definitions for each stage
        ch = [base_channels * (2 ** i) for i in range(stages)]
        
        # Initial Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoder Layers
        self.encoders = nn.ModuleList()
        for i in range(stages - 1):
            self.encoders.append(EncoderBlock(ch[i], ch[i+1], drop_rate))

        # Decoder Layers and Output Setup
        dec_out_ch = [ch[i-1] if i > 0 else ch[0] for i in range(stages)]
        self.decoders = nn.ModuleList([DecoderBlock(ch[i], dec_out_ch[i]) for i in range(stages)])

        # Adaptive Gating Components
        self.pre_fusion = nn.ModuleList([nn.Conv2d(ch[i], ch[i], 1, bias=False) for i in range(stages)])
        self.post_fusion = nn.ModuleList([nn.Conv2d(ch[i], dec_out_ch[i], 1, bias=False) for i in range(stages)])
        
        # Cross-stage connectivity logic
        self.targets = [i - 1 if i > 0 else (1 if stages > 1 else 0) for i in range(stages)]
        self.cross_fusion = nn.ModuleList([
            nn.Conv2d(ch[i], ch[self.targets[i]], 1, bias=False) for i in range(stages)
        ])

        # Learnable Gates Initialization
        if learnable_gating:
            # Initialize with bias: +5.0 (sigmoid ~1.0) for even stages (Res), -5.0 (sigmoid ~0.0) for odd (Skip)
            initial_gates = [5.0 if i % 2 == 0 else -5.0 for i in range(stages)]
            self.gates = nn.Parameter(torch.tensor(initial_gates))
        else:
            self.gates = None

        # Final Segmentation Head
        self.head = nn.Sequential(
            nn.Conv2d(ch[0], ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch[0], num_classes if num_classes > 1 else 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Encoder Pass
        x0 = self.stem(x)
        f0 = self.pool(x0)
        
        features = [f0]
        curr_feat = f0
        for enc in self.encoders:
            curr_feat = enc(curr_feat)
            features.append(curr_feat)
        
        # Ensure we process only the requested number of stages
        features = features[:self.stages]

        # 2. Compute Gating & Cross-Stage Features
        cross_stage_extras = [None] * self.stages
        residual_injections = [None] * self.stages

        for i in range(self.stages):
            # Calculate Gate Weight (w)
            if self.gates is not None:
                w = torch.sigmoid(self.gates[i]).view(1, 1, 1, 1)
            else:
                w = torch.ones(1, 1, 1, 1, device=features[i].device)
            
            complement_w = 1.0 - w
            target_idx = self.targets[i]

            # Cross-stage connection (weighted by complement_w)
            cross_feat = self.cross_fusion[i](features[i])
            target_shape = features[target_idx].shape[-2:]
            
            if cross_feat.shape[-2:] != target_shape:
                cross_feat = F.interpolate(cross_feat, size=target_shape, mode='bilinear', align_corners=False)
            
            cross_feat = cross_feat * complement_w
            
            # Accumulate cross-stage features
            if cross_stage_extras[target_idx] is None:
                cross_stage_extras[target_idx] = cross_feat
            else:
                cross_stage_extras[target_idx] = cross_stage_extras[target_idx] + cross_feat

            # Residual injection (weighted by w)
            res_feat = self.post_fusion[i](features[i]) * w
            residual_injections[i] = res_feat

        # 3. Decoder Pass with Fusion
        x_dec = None
        decoded_outputs = [None] * self.stages
        
        for i in reversed(range(self.stages)):
            y = features[i]
            
            # Add previous decoder output (standard U-Net path)
            if x_dec is not None:
                y = y + x_dec
            
            # Add pre-fusion and cross-stage extras
            y = y + self.pre_fusion[i](features[i])
            
            if cross_stage_extras[i] is not None:
                extra = cross_stage_extras[i]
                if extra.shape[-2:] != y.shape[-2:]:
                    extra = F.interpolate(extra, size=y.shape[-2:], mode='bilinear', align_corners=False)
                y = y + extra
            
            # Decode
            d = self.decoders[i](y)
            
            # Inject residual connection
            res = residual_injections[i]
            if res is not None:
                if res.shape[-2:] != d.shape[-2:]:
                    res = F.interpolate(res, size=d.shape[-2:], mode='bilinear', align_corners=False)
                d = d + res
            
            decoded_outputs[i] = d
            x_dec = d

        # 4. Final Classification
        out = self.head(decoded_outputs[0])
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        
        return out