"""
First Order Motion Model (FOMM) — complete PyTorch implementation.

Based on "First Order Motion Model for Image Animation" (Siarohin et al., NeurIPS 2019).
Architecture: Keypoint Detector → Dense Motion Network → Occlusion-Aware Generator.

The model animates a source (avatar) image by transferring motion from a driving
frame. Keypoints are detected in both images; the difference defines an optical
flow field used to warp the source and generate the animated output.

Default config matches the `vox-cpk.pth.tar` checkpoint (trained on VoxCeleb).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ━━━━━━━━━━━━━━━━━━━━ Utilities ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_coordinate_grid(spatial_size, device):
    """Create a 2D normalized coordinate grid in [-1, 1]."""
    h, w = spatial_size
    y = torch.arange(h, device=device).float()
    x = torch.arange(w, device=device).float()
    y = 2.0 * (y / (h - 1)) - 1.0
    x = 2.0 * (x / (w - 1)) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1)  # (H, W, 2)


def kp2gaussian(kp, spatial_size, kp_variance=0.01):
    """Convert keypoint coordinates to gaussian heatmaps."""
    mean = kp["value"]  # (B, K, 2)
    coordinate_grid = make_coordinate_grid(spatial_size, mean.device)  # (H, W, 2)
    # (B, K, 1, 1, 2) - (1, 1, H, W, 2)
    diff = coordinate_grid.view(1, 1, *spatial_size, 2) - mean.view(mean.shape[0], mean.shape[1], 1, 1, 2)
    out = torch.exp(-0.5 * (diff ** 2).sum(-1) / kp_variance)  # (B, K, H, W)
    return out


# ━━━━━━━━━━━━━━━━━━━━ Building Blocks ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AntiAliasInterpolation2d(nn.Module):
    """Gaussian blur before downsampling to prevent aliasing."""

    def __init__(self, channels, scale):
        super().__init__()
        sigma = (1.0 / scale - 1.0) / 2.0
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_1d = torch.exp(
            -torch.arange(-self.ka, self.ka + 1, dtype=torch.float32) ** 2
            / (2.0 * sigma ** 2)
        )
        kernel = (kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, x):
        if self.scale == 1.0:
            return x
        x = F.pad(x, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(x, self.weight, groups=self.groups)
        return F.interpolate(out, scale_factor=(self.scale, self.scale),
                             mode="bilinear", align_corners=False)


class SameBlock2d(nn.Module):
    """Conv + BatchNorm + ReLU preserving spatial dimensions."""

    def __init__(self, in_f, out_f, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_f, out_f, kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_f, affine=True)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class DownBlock2d(nn.Module):
    """Conv + BatchNorm + ReLU + AvgPool (halves spatial dims)."""

    def __init__(self, in_f, out_f, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_f, out_f, kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_f, affine=True)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(F.relu(self.norm(self.conv(x))))


class UpBlock2d(nn.Module):
    """Upsample + Conv + BatchNorm + ReLU (doubles spatial dims)."""

    def __init__(self, in_f, out_f, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_f, out_f, kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_f, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return F.relu(self.norm(self.conv(out)))


class ResBlock2d(nn.Module):
    """Two-conv residual block."""

    def __init__(self, in_f, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_f, in_f, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_f, in_f, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm2d(in_f, affine=True)
        self.norm2 = nn.BatchNorm2d(in_f, affine=True)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return x + out


# ━━━━━━━━━━━━━━━━━━━━ Hourglass (U-Net) ━━━━━━━━━━━━━━━━━━━━━━━━━

class Hourglass(nn.Module):
    """Symmetric encoder-decoder with skip connections."""

    def __init__(self, block_expansion, in_features, num_blocks=5, max_features=1024):
        super().__init__()
        # Encoder
        enc = []
        for i in range(num_blocks):
            in_f = in_features if i == 0 else min(max_features, block_expansion * (2 ** (i - 1)))
            out_f = min(max_features, block_expansion * (2 ** i))
            enc.append(DownBlock2d(in_f, out_f))
        self.encoder = nn.ModuleList(enc)

        # Decoder
        dec = []
        for i in range(num_blocks)[::-1]:
            encoder_out_f = min(max_features, block_expansion * (2 ** i))
            skip_f = in_features if i == 0 else min(max_features, block_expansion * (2 ** (i - 1)))
            out_f = skip_f if i > 0 else block_expansion
            dec.append(UpBlock2d(encoder_out_f + skip_f, out_f))
        self.decoder = nn.ModuleList(dec)

        self.out_filters = block_expansion + in_features

    def forward(self, x):
        skips = [x]
        for block in self.encoder:
            skips.append(block(skips[-1]))

        out = skips[-1]
        for i, block in enumerate(self.decoder):
            skip = skips[-(i + 2)]
            out = block(torch.cat([out, skip], dim=1))

        return torch.cat([out, x], dim=1)


# ━━━━━━━━━━━━━━━━━━━━ Keypoint Detector ━━━━━━━━━━━━━━━━━━━━━━━━━

class KPDetector(nn.Module):
    """
    Detects K abstract keypoints + local Jacobian affine transforms.
    Uses soft-argmax on predicted heatmaps for differentiable coordinates.
    """

    def __init__(self, block_expansion=32, num_kp=10, num_channels=3,
                 max_features=1024, num_blocks=5, temperature=0.1,
                 estimate_jacobian=True, scale_factor=0.25):
        super().__init__()
        self.predictor = Hourglass(block_expansion, num_channels, num_blocks, max_features)
        self.kp = nn.Conv2d(self.predictor.out_filters, num_kp, kernel_size=7, padding=3)
        self.num_kp = num_kp
        self.temperature = temperature
        self.scale_factor = scale_factor

        if estimate_jacobian:
            self.jacobian = nn.Conv2d(self.predictor.out_filters, 4 * num_kp,
                                      kernel_size=7, padding=3)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(
                torch.tensor([1, 0, 0, 1] * num_kp, dtype=torch.float32)
            )
        else:
            self.jacobian = None

        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, scale_factor)

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        # Soft-argmax: expected coordinate from heatmap
        grid = make_coordinate_grid(final_shape[2:], prediction.device)  # (H, W, 2)
        grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 2)
        value = (heatmap.unsqueeze(-1) * grid).sum(dim=(2, 3))  # (B, K, 2)
        out = {"value": value}

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(
                final_shape[0], self.num_kp, 4, final_shape[2], final_shape[3]
            )
            heatmap_rep = heatmap.unsqueeze(2)  # (B, K, 1, H, W)
            jacobian = (heatmap_rep * jacobian_map).sum(dim=(3, 4))  # (B, K, 4)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out["jacobian"] = jacobian

        return out


# ━━━━━━━━━━━━━━━━━━━━ Dense Motion Network ━━━━━━━━━━━━━━━━━━━━━━

class DenseMotionNetwork(nn.Module):
    """
    Converts sparse keypoint motion into a dense optical flow field.
    Produces per-pixel deformation + occlusion mask.
    """

    def __init__(self, block_expansion=64, num_blocks=5, max_features=1024,
                 num_kp=10, num_channels=3, estimate_occlusion_map=True,
                 scale_factor=0.25):
        super().__init__()
        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.estimate_occlusion_map = estimate_occlusion_map

        # Input: (num_kp+1) heatmaps + (num_kp+1) warped source channels
        in_features = (num_kp + 1) * (num_channels + 1)
        self.hourglass = Hourglass(block_expansion, in_features, num_blocks, max_features)
        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1,
                              kernel_size=7, padding=3)

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1,
                                       kernel_size=7, padding=3)
        else:
            self.occlusion = None

        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, scale_factor)

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), source_image.device)
        identity_grid = identity_grid.view(1, 1, h, w, 2)

        # Per-keypoint coordinate grids shifted to source space
        coordinate_grid = identity_grid - kp_driving["value"].view(bs, self.num_kp, 1, 1, 2)
        if "jacobian" in kp_driving:
            jac = torch.matmul(
                kp_source["jacobian"],
                torch.inverse(kp_driving["jacobian"])
            )
            coordinate_grid = torch.matmul(
                jac.unsqueeze(-3).unsqueeze(-3),
                coordinate_grid.unsqueeze(-1),
            ).squeeze(-1)
        coordinate_grid = coordinate_grid + kp_source["value"].view(bs, self.num_kp, 1, 1, 2)

        # Background motion = identity
        bg_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([bg_grid, coordinate_grid], dim=1)  # (B, K+1, H, W, 2)

        return sparse_motions

    def create_deformed_source(self, source_image, sparse_motions):
        bs, _, h, w = source_image.shape
        num_motions = sparse_motions.shape[1]  # K+1
        source_rep = source_image.unsqueeze(1).repeat(1, num_motions, 1, 1, 1)
        source_rep = source_rep.view(bs * num_motions, -1, h, w)
        grid = sparse_motions.view(bs * num_motions, h, w, 2)
        deformed = F.grid_sample(source_rep, grid, align_corners=False, padding_mode="border")
        return deformed.view(bs, num_motions, -1, h, w)

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size, kp_variance=0.01)
        gaussian_source = kp2gaussian(kp_source, spatial_size, kp_variance=0.01)
        heatmap = gaussian_driving - gaussian_source  # (B, K, H, W)
        zeros = torch.zeros(heatmap.shape[0], 1, *spatial_size, device=heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)  # (B, K+1, H, W)
        return heatmap.unsqueeze(2)  # (B, K+1, 1, H, W)

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        sparse_motions = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source(source_image, sparse_motions)
        heatmap = self.create_heatmap_representations(source_image, kp_driving, kp_source)

        # Concatenate deformed sources with heatmaps and reshape
        inp = torch.cat([heatmap, deformed_source], dim=2)  # (B, K+1, C+1, H, W)
        inp = inp.view(bs, -1, h, w)

        prediction = self.hourglass(inp)

        # Per-keypoint soft mask
        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1).unsqueeze(2)  # (B, K+1, 1, H, W)

        # Weighted combination of sparse motions
        sparse_motions = sparse_motions.permute(0, 1, 4, 2, 3)  # (B, K+1, 2, H, W)
        deformation = (mask * sparse_motions).sum(dim=1)  # (B, 2, H, W)
        deformation = deformation.permute(0, 2, 3, 1)  # (B, H, W, 2)

        out = {"deformation": deformation}

        if self.occlusion is not None:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out["occlusion_map"] = occlusion_map

        return out


# ━━━━━━━━━━━━━━━━━━━━ Generator ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OcclusionAwareGenerator(nn.Module):
    """
    Warps encoded source features using the dense motion field,
    applies occlusion masking, and decodes to the output image.
    """

    def __init__(self, num_channels=3, num_kp=10, block_expansion=64,
                 max_features=512, num_down_blocks=2, num_bottleneck_blocks=6,
                 estimate_occlusion_map=True, dense_motion_params=None):
        super().__init__()
        dm_params = dense_motion_params or {}
        self.dense_motion_network = DenseMotionNetwork(
            num_kp=num_kp, num_channels=num_channels,
            estimate_occlusion_map=estimate_occlusion_map,
            **dm_params,
        )

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=7, padding=3)

        down = []
        for i in range(num_down_blocks):
            in_f = min(max_features, block_expansion * (2 ** i))
            out_f = min(max_features, block_expansion * (2 ** (i + 1)))
            down.append(DownBlock2d(in_f, out_f))
        self.down_blocks = nn.ModuleList(down)

        up = []
        for i in range(num_down_blocks)[::-1]:
            in_f = min(max_features, block_expansion * (2 ** (i + 1)))
            out_f = min(max_features, block_expansion * (2 ** i))
            up.append(UpBlock2d(in_f, out_f))
        self.up_blocks = nn.ModuleList(up)

        bottleneck = []
        in_f = min(max_features, block_expansion * (2 ** num_down_blocks))
        for _ in range(num_bottleneck_blocks):
            bottleneck.append(ResBlock2d(in_f))
        self.bottleneck = nn.Sequential(*bottleneck)

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=7, padding=3)
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        """Warp input features using the deformation grid."""
        _, _, h_old, w_old = inp.shape
        _, h_new, w_new, _ = deformation.shape
        if h_old != h_new or w_old != w_new:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h_old, w_old),
                                         mode="bilinear", align_corners=False)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, align_corners=False, padding_mode="border")

    def forward(self, source_image, kp_driving, kp_source):
        out = self.first(source_image)
        for block in self.down_blocks:
            out = block(out)

        # Dense motion estimation
        dense_motion = self.dense_motion_network(
            source_image=source_image,
            kp_driving=kp_driving,
            kp_source=kp_source,
        )
        occlusion_map = dense_motion.get("occlusion_map")
        deformation = dense_motion["deformation"]

        # Warp encoded features
        out = self.deform_input(out, deformation)

        # Apply occlusion
        if occlusion_map is not None:
            if out.shape[2:] != occlusion_map.shape[2:]:
                occlusion_map = F.interpolate(
                    occlusion_map, size=out.shape[2:],
                    mode="bilinear", align_corners=False
                )
            out = out * occlusion_map

        # Bottleneck
        out = self.bottleneck(out)

        # Decode
        for block in self.up_blocks:
            out = block(out)

        out = self.final(out)
        out = torch.sigmoid(out)

        return {"prediction": out}


# ━━━━━━━━━━━━━━━━━━━━ Full FOMM Model ━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FOMM(nn.Module):
    """
    Complete First Order Motion Model.
    Combines KPDetector + OcclusionAwareGenerator.
    """

    def __init__(self, num_channels=3, num_kp=10, kp_detector_params=None,
                 generator_params=None):
        super().__init__()
        kp_params = kp_detector_params or {}
        gen_params = generator_params or {}

        self.kp_detector = KPDetector(
            num_kp=num_kp, num_channels=num_channels, **kp_params
        )
        self.generator = OcclusionAwareGenerator(
            num_kp=num_kp, num_channels=num_channels, **gen_params
        )

    def forward(self, source_image, driving_frame):
        kp_source = self.kp_detector(source_image)
        kp_driving = self.kp_detector(driving_frame)
        generated = self.generator(source_image, kp_driving=kp_driving, kp_source=kp_source)
        return generated

    def animate(self, source_image, kp_source, kp_driving):
        """Animate using pre-extracted keypoints (avoids re-detecting source KP)."""
        return self.generator(source_image, kp_driving=kp_driving, kp_source=kp_source)


def build_fomm(checkpoint_path: Optional[str] = None, device: str = "cpu") -> FOMM:
    """
    Build FOMM with the standard vox-cpk configuration and optionally
    load pretrained weights.
    """
    model = FOMM(
        num_channels=3,
        num_kp=10,
        kp_detector_params={
            "block_expansion": 32,
            "max_features": 1024,
            "num_blocks": 5,
            "temperature": 0.1,
            "estimate_jacobian": True,
            "scale_factor": 0.25,
        },
        generator_params={
            "block_expansion": 64,
            "max_features": 512,
            "num_down_blocks": 2,
            "num_bottleneck_blocks": 6,
            "estimate_occlusion_map": True,
            "dense_motion_params": {
                "block_expansion": 64,
                "max_features": 1024,
                "num_blocks": 5,
                "scale_factor": 0.25,
            },
        },
    )

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "kp_detector" in checkpoint:
            model.kp_detector.load_state_dict(checkpoint["kp_detector"])
            model.generator.load_state_dict(checkpoint["generator"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model
