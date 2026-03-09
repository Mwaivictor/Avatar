"""
Wav2Lip — Speech-Driven Lip Sync Model (complete PyTorch implementation).

Based on "A Lip Sync Expert Is All You Need for Speech to Lip Generation"
(Prajwal et al., ACM Multimedia 2020).

Architecture:
  - Face encoder:  Encodes the masked lower-half face image through
                   a series of residual conv blocks, producing face embeddings.
  - Audio encoder: Encodes mel spectrogram of speech through conv blocks,
                   producing audio embeddings.
  - Face decoder:  Fuses audio + face embeddings via transposed convolutions
                   to generate the lip-synced face region.

Input:
  - Face images:  (B, 6, 96, 96) — 2 concatenated 3-channel images:
                  reference face (full) + target face (lower half masked).
  - Audio:        (B, 1, 80, 16) — mel spectrogram (80 mel bins, 16 time steps).

Output:
  - Synced face:  (B, 3, 96, 96) — lip-synced face image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━ Conv Blocks ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Conv2dBlock(nn.Module):
    """Conv2D + BatchNorm + ReLU with optional residual connection."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 residual=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, x):
        out = self.conv(x)
        if self.residual:
            out = out + x
        return self.act(out)


class ConvTranspose2dBlock(nn.Module):
    """ConvTranspose2D + BatchNorm + ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding,
                 output_padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding,
                               output_padding=output_padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ━━━━━━━━━━━━━━━━━━━━ Audio Encoder ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AudioEncoder(nn.Module):
    """
    Encodes mel spectrograms into audio embeddings.
    Input: (B, 1, 80, 16) → Output: (B, 512, 1, 1)
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Conv2dBlock(1, 32, 3, 1, 1),
            Conv2dBlock(32, 32, 3, 1, 1, residual=True),
            Conv2dBlock(32, 32, 3, 1, 1, residual=True),

            Conv2dBlock(32, 64, 3, stride=(3, 1), padding=1),      # 80→27, 16→16
            Conv2dBlock(64, 64, 3, 1, 1, residual=True),
            Conv2dBlock(64, 64, 3, 1, 1, residual=True),

            Conv2dBlock(64, 128, 3, stride=3, padding=1),           # 27→9, 16→6
            Conv2dBlock(128, 128, 3, 1, 1, residual=True),
            Conv2dBlock(128, 128, 3, 1, 1, residual=True),

            Conv2dBlock(128, 256, 3, stride=(3, 2), padding=1),     # 9→3, 6→3
            Conv2dBlock(256, 256, 3, 1, 1, residual=True),

            Conv2dBlock(256, 512, 3, 1, padding=0),                 # 3→1, 3→1
            Conv2dBlock(512, 512, 1, 1, padding=0),
        )

    def forward(self, x):
        return self.layers(x)


# ━━━━━━━━━━━━━━━━━━━━ Face Encoder ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FaceEncoder(nn.Module):
    """
    Encodes face images into face embeddings.
    Input: (B, 6, 96, 96) → Output: (B, 512, 3, 3)

    Face input is the concatenation of:
      - Reference face (3 channels, full face)
      - Target face (3 channels, lower half zeroed as mask)
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                Conv2dBlock(6, 16, 7, 1, 3),                                     # 96×96
            ),
            nn.Sequential(
                Conv2dBlock(16, 32, 3, 2, 1),                                    # 48×48
                Conv2dBlock(32, 32, 3, 1, 1, residual=True),
                Conv2dBlock(32, 32, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dBlock(32, 64, 3, 2, 1),                                    # 24×24
                Conv2dBlock(64, 64, 3, 1, 1, residual=True),
                Conv2dBlock(64, 64, 3, 1, 1, residual=True),
                Conv2dBlock(64, 64, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dBlock(64, 128, 3, 2, 1),                                   # 12×12
                Conv2dBlock(128, 128, 3, 1, 1, residual=True),
                Conv2dBlock(128, 128, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dBlock(128, 256, 3, 2, 1),                                  # 6×6
                Conv2dBlock(256, 256, 3, 1, 1, residual=True),
                Conv2dBlock(256, 256, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dBlock(256, 512, 3, 2, 1),                                  # 3×3
                Conv2dBlock(512, 512, 3, 1, 1, residual=True),
            ),
        ])

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


# ━━━━━━━━━━━━━━━━━━━━ Face Decoder ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FaceDecoder(nn.Module):
    """
    Decodes fused audio+face features back to face image with synced lips.
    Input: (B, 512, 3, 3) from face encoder + (B, 512, 1, 1) from audio encoder.
    Output: (B, 3, 96, 96)
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                ConvTranspose2dBlock(1024, 512, 3, 1, 0),                        # 3→5
                Conv2dBlock(512, 512, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                ConvTranspose2dBlock(1024, 512, 3, 2, 1, output_padding=1),      # 5→10... but with skip
                Conv2dBlock(512, 512, 3, 1, 1, residual=True),
                Conv2dBlock(512, 512, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                ConvTranspose2dBlock(768, 384, 3, 2, 1, output_padding=1),       # with 256 skip
                Conv2dBlock(384, 384, 3, 1, 1, residual=True),
                Conv2dBlock(384, 384, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                ConvTranspose2dBlock(512, 256, 3, 2, 1, output_padding=1),       # with 128 skip
                Conv2dBlock(256, 256, 3, 1, 1, residual=True),
                Conv2dBlock(256, 256, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                ConvTranspose2dBlock(320, 128, 3, 2, 1, output_padding=1),       # with 64 skip
                Conv2dBlock(128, 128, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                ConvTranspose2dBlock(160, 64, 3, 2, 1, output_padding=1),        # with 32 skip
                Conv2dBlock(64, 64, 3, 1, 1, residual=True),
            ),
        ])
        self.output_conv = nn.Sequential(
            nn.Conv2d(80, 3, 1, 1, 0),  # 64 + 16 skip
            nn.Sigmoid(),
        )

    def forward(self, audio_feat, face_feats):
        """
        Args:
            audio_feat: (B, 512, 1, 1) from audio encoder.
            face_feats: List of face encoder features at each scale.
        """
        # Broadcast audio embedding to match face spatial dims
        x = face_feats[-1]  # (B, 512, 3, 3)
        audio_expanded = audio_feat.expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, audio_expanded], dim=1)  # (B, 1024, 3, 3)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                # Skip connection from face encoder (reverse order)
                skip = face_feats[-(i + 2)]
                # Ensure spatial dimensions match
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                                      align_corners=False)
                x = torch.cat([x, skip], dim=1)

        # Final skip from first encoder layer
        skip = face_feats[0]
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.output_conv(x)
        return x


# ━━━━━━━━━━━━━━━━━━━━ Complete Wav2Lip Model ━━━━━━━━━━━━━━━━━━━━

class Wav2Lip(nn.Module):
    """
    Complete Wav2Lip model combining audio encoder, face encoder, and
    face decoder for speech-driven lip synchronization.
    """

    def __init__(self):
        super().__init__()
        self.face_encoder = FaceEncoder()
        self.audio_encoder = AudioEncoder()
        self.face_decoder = FaceDecoder()

    def forward(self, audio_sequences, face_sequences):
        """
        Args:
            audio_sequences: (B, 1, 80, 16) mel spectrogram.
            face_sequences: (B, 6, 96, 96) concatenated face images.

        Returns:
            (B, 3, 96, 96) lip-synced face image.
        """
        audio_embedding = self.audio_encoder(audio_sequences)
        face_features = self.face_encoder(face_sequences)
        output = self.face_decoder(audio_embedding, face_features)
        return output


def build_wav2lip(checkpoint_path: Optional[str] = None,
                  device: str = "cpu") -> Wav2Lip:
    """Build Wav2Lip with optional pretrained weights."""
    model = Wav2Lip()

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device,
                                weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Handle 'module.' prefix from DataParallel training
        cleaned = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            cleaned[name] = v

        model.load_state_dict(cleaned, strict=False)

    model = model.to(device)
    model.eval()
    return model
