import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class CCG(nn.Module):
    """
    Clutter-suppressing Cross-scale Gating (CCG) Module.
    
    Filters shallow encoder background noise using deep semantic decoder features as anchors
    via joint statistical channel-wise gating.
    
    Args:
        channel (int): Base channel dimension of encoder feature map (xe).
        reduction (int): Channel reduction factor for 1x1 grouped convolutions. Default: 16.
        groups (int): Group number for grouped convolutions. Default: 4.
    """
    def __init__(self, channel: int, reduction: int = 16, groups: int = 4):
        super().__init__()
        self.C = channel
        in_ch = 3 * channel
        hid_ch = max(1, in_ch // reduction)
        out_ch = 3 * channel

        def best_groups(max_groups: int, a: int, b: int) -> int:
            g = min(max_groups, a, b)
            while g > 1 and not (a % g == 0 and b % g == 0):
                g -= 1
            return g

        g1 = best_groups(groups, in_ch, hid_ch)
        g2 = best_groups(groups, hid_ch, out_ch)

        self.conv_head = nn.Conv2d(in_ch, hid_ch, kernel_size=1, groups=g1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.conv_tail = nn.Conv2d(hid_ch, out_ch, kernel_size=1, groups=g2, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CCG module.
        x1: Encoder feature map (B, C, H, W)
        x2: Upsampled decoder feature map (B, 2C, H, W)
        """
        B, C, H, W = x1.shape
        assert x2.size(1) == 2 * C, f"x2 channel count expected {2*C}, got {x2.size(1)}"
        x = torch.cat([x1, x2], dim=1)                     # (B, 3C, H, W)
        gap = F.adaptive_avg_pool2d(x, 1)                  # (B, 3C, 1, 1)
        y = self.conv_tail(self.act(self.conv_head(gap)))  # (B, 3C, 1, 1)

        w1 = torch.sigmoid(y[:, :C])
        w2 = torch.sigmoid(y[:, C:])
        out1 = x1 * w1
        out2 = x2 * w2
        return torch.cat([out1, out2], dim=1)


class HaarWaveletDownsample(nn.Module):
    """
    Single-level 2D Discrete Haar Wavelet Transform (DWT) Downsampler.
    Decomposes feature maps into low-frequency (LL) and high-frequency (HL, LH, HH) sub-bands.
    """
    def __init__(self, in_ch: int, out_ch: int, wave: str = 'haar', mode: str = 'symmetric'):
        super().__init__()
        self.wt = DWTForward(J=1, mode=mode, wave=wave)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        yL, yH = self.wt(x)  # yL: [N, C, H/2, W/2], yH[0]: [N, C, 3, H/2, W/2]
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]
        x_concat = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        return self.proj(x_concat)


class HPDFusionGate(nn.Module):
    """
    Channel-wise Scaling Gate for HPD.
    Dynamically re-weights features from MaxPool branch and DWT branch.
    """
    def __init__(self, channel: int, reduction: int = 16, groups: int = 4):
        super().__init__()
        C = channel
        in_ch_att = 2 * C
        hid_ch = max(1, in_ch_att // reduction)
        out_ch_att = 2 * C

        def best_groups(max_groups: int, a: int, b: int) -> int:
            g = min(max_groups, a, b)
            while g > 1 and not (a % g == 0 and b % g == 0):
                g -= 1
            return g

        g1 = best_groups(groups, in_ch_att, hid_ch)
        g2 = best_groups(groups, hid_ch, out_ch_att)

        self.conv_head = nn.Conv2d(in_ch_att, hid_ch, kernel_size=1, groups=g1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.conv_tail = nn.Conv2d(hid_ch, out_ch_att, kernel_size=1, groups=g2, bias=False)

        self.fuse = nn.Sequential(
            nn.Conv2d(2 * C, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=1)
        gap = F.adaptive_avg_pool2d(x, 1)
        y = self.conv_tail(self.act(self.conv_head(gap)))

        C = x1.shape[1]
        w1 = torch.sigmoid(y[:, :C])
        w2 = torch.sigmoid(y[:, C:])
        out1 = x1 * w1
        out2 = x2 * w2
        return self.fuse(torch.cat([out1, out2], dim=1))


class HPD(nn.Module):
    """
    High-frequency Preserving Downsampler (HPD).
    
    Combines spatial MaxPool (translation invariance) with Haar Wavelet decomposition
    (high-frequency spectral energy preservation) to prevent target energy dilution during spatial compression.
    """
    def __init__(self, in_ch: int, out_ch: int, use_gate: bool = True, wave: str = 'haar', mode: str = 'symmetric'):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.hwd = HaarWaveletDownsample(in_ch, out_ch, wave=wave, mode=mode)
        self.use_gate = use_gate

        if use_gate:
            self.gate = HPDFusionGate(out_ch)
        else:
            self.fuse = nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pool = self.pool_proj(self.pool(x))
        x_hwd = self.hwd(x)
        if self.use_gate:
            return self.gate(x_pool, x_hwd)
        else:
            return self.fuse(torch.cat([x_pool, x_hwd], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, max(1, in_planes // ratio), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(max(1, in_planes // ratio), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))


class ResBlock(nn.Module):
    """
    Residual Block with Channel and Spatial Attention.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        return self.relu(out)


class UpsampleDeterministic(nn.Module):
    """
    Deterministic Upsampling operator via pixel expansion (avoids interpolation blurring).
    """
    def __init__(self, upscale: int = 2):
        super().__init__()
        self.upscale = upscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :, None, :, None]\
            .expand(-1, -1, -1, self.upscale, -1, self.upscale)\
            .reshape(x.size(0), x.size(1), x.size(2) * self.upscale, x.size(3) * self.upscale)


class EdgeRefine(nn.Module):
    """
    Sobel-based Edge Gradient Refinement Operator.
    Uses fixed 2D Sobel operators (Gx, Gy) depthwise convolution to extract spatial gradient magnitude,
    projecting it to a spatial weight mask for residual geometric boundary enhancement.
    """
    def __init__(self, in_ch: int, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.gx = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False, groups=in_ch)
        self.gy = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False, groups=in_ch)

        with torch.no_grad():
            kx = torch.tensor([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]], dtype=torch.float32)
            ky = kx.t()

            w_x = torch.zeros(in_ch, 1, 3, 3)
            w_y = torch.zeros(in_ch, 1, 3, 3)
            w_x[:, 0, :, :] = kx
            w_y[:, 0, :, :] = ky

            self.gx.weight.copy_(w_x)
            self.gy.weight.copy_(w_y)

        for p in self.gx.parameters():
            p.requires_grad = False
        for p in self.gy.parameters():
            p.requires_grad = False

        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            norm_layer(in_ch),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.gx(x)
        gy = self.gy(x)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        att = self.fuse(mag)
        return x * (1.0 + att)


class BGR(nn.Module):
    """
    Boundary-guided Geometric Restorer (BGR).
    Integrates spatial resolution expansion with fixed Sobel gradient prior constraints.
    """
    def __init__(self, in_ch: int, upscale: int = 2, order: str = 'post', norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert order in ('post', 'pre')
        self.order = order
        self.ups = UpsampleDeterministic(upscale=upscale)
        self.edge = EdgeRefine(in_ch=in_ch, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.order == 'pre':
            x = self.edge(x)
            x = self.ups(x)
        else:
            x = self.ups(x)
            x = self.edge(x)
        return x


class SEG_UNet(nn.Module):
    """
    SEG-UNet: Spectral-Edge Gated U-Net for Infrared Small-Target Segmentation.
    
    Paper Architecture Components:
      1. High-frequency Preserving Downsampler (HPD) in Encoder
      2. Clutter-suppressing Cross-scale Gating (CCG) in Skip-Connections
      3. Boundary-guided Geometric Restorer (BGR) in Decoder & Auxiliary Heads
    """
    def __init__(self, input_channels: int = 1, block=ResBlock):
        super().__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]

        # Encoder HPD Downsamplers
        self.conv_init = nn.Conv2d(input_channels, param_channels[0], kernel_size=1, stride=1)
        self.down_wt_1 = HPD(param_channels[0], param_channels[1], use_gate=True)
        self.down_wt_2 = HPD(param_channels[1], param_channels[2], use_gate=True)
        self.down_wt_3 = HPD(param_channels[2], param_channels[3], use_gate=True)
        self.down_wt_4 = HPD(param_channels[3], param_channels[4], use_gate=True)

        # Encoder Layers
        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[1], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[2], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[3], param_channels[3], block, param_blocks[2])
        self.middle_layer = self._make_layer(param_channels[4], param_channels[4], block, param_blocks[3])

        # Skip-Connection CCG Modules
        self.middle_3 = CCG(param_channels[3])
        self.middle_2 = CCG(param_channels[2])
        self.middle_1 = CCG(param_channels[1])
        self.middle_0 = CCG(param_channels[0])

        # Decoder BGR Upsamplers
        self.up_m = BGR(in_ch=param_channels[4], upscale=2, order='post')
        self.up_d3 = BGR(in_ch=param_channels[3], upscale=2, order='post')
        self.up_d2 = BGR(in_ch=param_channels[2], upscale=2, order='post')
        self.up_d1 = BGR(in_ch=param_channels[1], upscale=2, order='post')

        # Decoder Layers
        self.decoder_3 = self._make_layer(param_channels[3] + param_channels[4], param_channels[3], block, param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[2] + param_channels[3], param_channels[2], block, param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[1] + param_channels[2], param_channels[1], block, param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[0] + param_channels[1], param_channels[0], block)

        # Multi-scale Deep Supervision Prediction Heads
        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.up_mask1 = BGR(in_ch=1, upscale=2, order='post')
        self.up_mask2 = BGR(in_ch=1, upscale=4, order='post')
        self.up_mask3 = BGR(in_ch=1, upscale=8, order='post')

        self.final = nn.Conv2d(4, 1, 3, 1, 1)

    def _make_layer(self, in_channels: int, out_channels: int, block, block_num: int = 1):
        layers = [block(in_channels, out_channels)]
        for _ in range(block_num - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, warm_flag: bool = True):
        """
        Forward pass.
        warm_flag: True during training (returns multi-scale auxiliary heads),
                   False during inference (returns primary prediction only).
        """
        # Encoder with HPD
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.down_wt_1(x_e0))
        x_e2 = self.encoder_2(self.down_wt_2(x_e1))
        x_e3 = self.encoder_3(self.down_wt_3(x_e2))
        x_m = self.middle_layer(self.down_wt_4(x_e3))

        # Decoder with CCG and BGR
        middle_x_3 = self.middle_3(x_e3, self.up_m(x_m))
        x_d3 = self.decoder_3(middle_x_3)
        
        middle_x_2 = self.middle_2(x_e2, self.up_d3(x_d3))
        x_d2 = self.decoder_2(middle_x_2)
        
        middle_x_1 = self.middle_1(x_e1, self.up_d2(x_d2))
        x_d1 = self.decoder_1(middle_x_1)
        
        middle_x_0 = self.middle_0(x_e0, self.up_d1(x_d1))
        x_d0 = self.decoder_0(middle_x_0)

        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)

            output = self.final(torch.cat([
                mask0,
                self.up_mask1(mask1),
                self.up_mask2(mask2),
                self.up_mask3(mask3)
            ], dim=1))
            return [mask0, mask1, mask2, mask3], output
        else:
            output = self.output_0(x_d0)
            return [], output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = 1
    batch_size = 1
    height, width = 256, 256
    x = torch.randn(batch_size, input_channels, height, width).to(device)

    model = SEG_UNet(input_channels).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SEG-UNet Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    try:
        from thop import profile
        flops, params = profile(model, inputs=(x, True), verbose=False)
        print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
    except ImportError:
        print("thop library not found. Install via 'pip install thop' to compute FLOPs.")

    masks, output = model(x, warm_flag=True)
    print(f"Output prediction shape: {output.shape}")