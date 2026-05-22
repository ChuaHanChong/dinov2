"""Real-ESRGAN degradation as a per-image augmentation for DINOv2 SSL.

Degradation params come unchanged from finetune_realesrgan_x4plus.yml. Two
deviations: `scale=2` (vs 4) so PIL output keeps input shape for the geometric
crops, and `use_usm_on_gt=False` (vs True) so the SSL pair is
clean <-> degrade(clean), not clean <-> degrade(USM(clean)).
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_DEGRADE_DIR = Path(__file__).resolve().parents[3] / "Real-ESRGAN" / "scripts"
if str(_DEGRADE_DIR) not in sys.path:
    sys.path.insert(0, str(_DEGRADE_DIR))
from degrade_dataset import (DegradationConfig, Degrader,  # noqa: E402
                             sample_kernels_for_image)


# 224 / sqrt(0.32) ≈ 396, the smallest source side that lets RandomResizedCrop
# produce a 224 global at min area scale 0.32 without first upsampling.
_DEFAULT_PREP_SHORT_SIDE = 400


class ESRGANDegradeTransform:
    """Roll Real-ESRGAN degradation on a PIL image with probability p."""

    def __init__(
        self,
        p: float = 0.8,
        scale: int = 2,
        restore_pre_degrade_size: bool = True,
        prep_short_side: Optional[int] = _DEFAULT_PREP_SHORT_SIDE,
        device: str = "cpu",
    ):
        self.p = float(p)
        self.scale = int(scale)
        self.restore = bool(restore_pre_degrade_size)
        self.prep_short_side = prep_short_side
        self.device_str = device
        # Built lazily inside the worker to avoid CUDA-init at fork.
        self._cfg: Optional[DegradationConfig] = None
        self._degrader: Optional[Degrader] = None

    def _lazy_init(self) -> None:
        if self._degrader is None:
            self._cfg = DegradationConfig(scale=self.scale, use_usm_on_gt=False)
            self._degrader = Degrader(self._cfg, torch.device(self.device_str))

    def __call__(self, image: Image.Image) -> Image.Image:
        if np.random.uniform() >= self.p:
            return image
        self._lazy_init()
        assert self._cfg is not None and self._degrader is not None

        if self.prep_short_side is not None:
            w, h = image.size
            short = min(w, h)
            if short > self.prep_short_side:
                ratio = self.prep_short_side / short
                new_size = (max(1, int(round(w * ratio))),
                            max(1, int(round(h * ratio))))
                image = image.resize(new_size, Image.BICUBIC)

        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        gt = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
        in_h, in_w = gt.shape[2:4]

        k1, k2, sinc = sample_kernels_for_image(self._cfg)
        out = self._degrader.degrade(
            gt, k1.unsqueeze(0), k2.unsqueeze(0), sinc.unsqueeze(0)
        )

        if self.restore and out.shape[2:4] != (in_h, in_w):
            out = F.interpolate(out, size=(in_h, in_w), mode="bicubic",
                                align_corners=False)
        out = torch.clamp(out, 0, 1)

        rgb_out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        arr = np.clip(rgb_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def __repr__(self) -> str:
        return (f"ESRGANDegradeTransform(p={self.p}, scale={self.scale}, "
                f"restore={self.restore}, prep_short_side={self.prep_short_side})")
