import argparse
import os
import random
import time
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

try:
    import kornia as K
    import kornia.feature as KF
except ImportError as exc:
    raise ImportError("kpfuse_v2.py requires kornia. Install with: pip install kornia") from exc


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def normalize_map_per_sample(x: torch.Tensor) -> torch.Tensor:
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-6)


def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] != 3:
        raise ValueError(f"Expected RGB tensor with 3 channels, got {x.shape}")
    return (0.299 * x[:, 0:1]) + (0.587 * x[:, 1:2]) + (0.114 * x[:, 2:3])


def gradient_mag_map(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt((dx * dx) + (dy * dy) + 1e-6)


def local_contrast_map(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size for local_contrast_map must be odd.")
    pad = kernel_size // 2
    mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    mean_sq = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=pad)
    var = (mean_sq - (mean * mean)).clamp_min(0.0)
    return torch.sqrt(var + 1e-6)


def _as_odd_kernel(kernel_size: int) -> int:
    k = max(1, int(kernel_size))
    return k if (k % 2 == 1) else (k + 1)


def smooth_spatial_map(
    x: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.2,
    passes: int = 1,
    mode: str = "gaussian",
) -> torch.Tensor:
    if kernel_size <= 1 or passes <= 0:
        return x
    k = _as_odd_kernel(kernel_size)
    out = x
    for _ in range(passes):
        if mode == "gaussian":
            out = K.filters.gaussian_blur2d(out, (k, k), (float(sigma), float(sigma)))
        elif mode == "box":
            out = F.avg_pool2d(out, kernel_size=k, stride=1, padding=k // 2)
        else:
            raise ValueError(f"Unsupported smooth mode: {mode}")
    return out


def apply_tensor_photometric(
    img: torch.Tensor,
    brightness_jitter: float,
    contrast_jitter: float,
) -> torch.Tensor:
    out = img
    if brightness_jitter > 0:
        b = random.uniform(max(0.0, 1.0 - brightness_jitter), 1.0 + brightness_jitter)
        out = TF.adjust_brightness(out, b)
    if contrast_jitter > 0:
        c = random.uniform(max(0.0, 1.0 - contrast_jitter), 1.0 + contrast_jitter)
        out = TF.adjust_contrast(out, c)
    return out


# =========================================================
# DATA
# =========================================================
def get_all_images(folder: str):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = []
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() in exts and p.is_file():
            files.append(p)
    return files


def match_triplets(vis_dir: str, ir_dir: str, gt_dir: str):
    vis = {p.name: p for p in get_all_images(vis_dir)}
    ir = {p.name: p for p in get_all_images(ir_dir)}
    gt = {p.name: p for p in get_all_images(gt_dir)}
    names = sorted(set(vis) & set(ir) & set(gt))
    return [(str(vis[n]), str(ir[n]), str(gt[n])) for n in names]


class FusionDataset(Dataset):
    def __init__(
        self,
        samples,
        size: int = 256,
        augment: bool = False,
        native_res: bool = False,
        max_side: int = 0,
        vis_brightness_jitter: float = 0.20,
        vis_contrast_jitter: float = 0.20,
        ir_brightness_jitter: float = 0.20,
        ir_contrast_jitter: float = 0.20,
        gt_brightness_jitter: float = 0.10,
        gt_contrast_jitter: float = 0.10,
        indices: Optional[Sequence[int]] = None,
    ):
        self.samples = list(samples) if indices is None else [samples[j] for j in indices]
        if len(self.samples) < 1:
            raise RuntimeError("Dataset split has no samples.")
        self.augment = augment
        self.native_res = native_res
        self.max_side = max_side
        self.resize = T.Resize((size, size))
        self.to_tensor = T.ToTensor()
        self.vis_brightness_jitter = vis_brightness_jitter
        self.vis_contrast_jitter = vis_contrast_jitter
        self.ir_brightness_jitter = ir_brightness_jitter
        self.ir_contrast_jitter = ir_contrast_jitter
        self.gt_brightness_jitter = gt_brightness_jitter
        self.gt_contrast_jitter = gt_contrast_jitter

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _sample_factor(jitter: float) -> float:
        if jitter <= 0:
            return 1.0
        lo = max(0.0, 1.0 - jitter)
        hi = 1.0 + jitter
        return random.uniform(lo, hi)

    def _apply_photometric(self, img: Image.Image, b_jitter: float, c_jitter: float):
        if b_jitter > 0:
            img = TF.adjust_brightness(img, self._sample_factor(b_jitter))
        if c_jitter > 0:
            img = TF.adjust_contrast(img, self._sample_factor(c_jitter))
        return img

    def _maybe_resize_native(
        self, v_img: Image.Image, i_img: Image.Image, g_img: Image.Image
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        if self.max_side <= 0:
            return v_img, i_img, g_img
        w, h = v_img.size
        cur_side = max(h, w)
        if cur_side <= self.max_side:
            return v_img, i_img, g_img
        scale = self.max_side / float(cur_side)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        out_size = [new_h, new_w]
        v_img = TF.resize(v_img, out_size)
        i_img = TF.resize(i_img, out_size)
        g_img = TF.resize(g_img, out_size)
        return v_img, i_img, g_img

    def __getitem__(self, i):
        vp, ip, gp = self.samples[i]
        with Image.open(vp) as v_img, Image.open(ip) as i_img, Image.open(gp) as g_img:
            v_img = v_img.convert("RGB")
            i_img = i_img.convert("L")
            g_img = g_img.convert("RGB")
            if self.native_res:
                v_img, i_img, g_img = self._maybe_resize_native(v_img, i_img, g_img)
            else:
                v_img = self.resize(v_img)
                i_img = self.resize(i_img)
                g_img = self.resize(g_img)

            if self.augment:
                if random.random() < 0.5:
                    v_img = TF.hflip(v_img)
                    i_img = TF.hflip(i_img)
                    g_img = TF.hflip(g_img)
                if random.random() < 0.2:
                    v_img = TF.vflip(v_img)
                    i_img = TF.vflip(i_img)
                    g_img = TF.vflip(g_img)

                v_img = self._apply_photometric(v_img, self.vis_brightness_jitter, self.vis_contrast_jitter)
                i_img = self._apply_photometric(i_img, self.ir_brightness_jitter, self.ir_contrast_jitter)
                g_img = self._apply_photometric(g_img, self.gt_brightness_jitter, self.gt_contrast_jitter)

            v = self.to_tensor(v_img)
            ir = self.to_tensor(i_img)
            gt = self.to_tensor(g_img)
        return v, ir, gt


def _pad_image_and_mask(
    img: torch.Tensor, target_h: int, target_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    _, h, w = img.shape
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Target padding size must be >= current tensor size.")
    pad = (0, pad_w, 0, pad_h)
    if pad_h > 0 or pad_w > 0:
        # Replicate padding avoids hard dark borders that can inject patch artifacts.
        img = F.pad(img, pad, mode="replicate")
    mask = torch.ones((1, h, w), dtype=img.dtype)
    mask = F.pad(mask, pad, mode="constant", value=0.0)
    return img, mask


def native_pad_collate(batch, pad_multiple: int = 8):
    vs, is_, gts = zip(*batch)
    max_h = max(v.shape[1] for v in vs)
    max_w = max(v.shape[2] for v in vs)
    if pad_multiple > 1:
        max_h = ((max_h + pad_multiple - 1) // pad_multiple) * pad_multiple
        max_w = ((max_w + pad_multiple - 1) // pad_multiple) * pad_multiple

    v_out, i_out, gt_out, m_out = [], [], [], []
    for v, i, gt in zip(vs, is_, gts):
        v_pad, mask = _pad_image_and_mask(v, max_h, max_w)
        i_pad, _ = _pad_image_and_mask(i, max_h, max_w)
        gt_pad, _ = _pad_image_and_mask(gt, max_h, max_w)
        v_out.append(v_pad)
        i_out.append(i_pad)
        gt_out.append(gt_pad)
        m_out.append(mask)
    return (
        torch.stack(v_out, dim=0),
        torch.stack(i_out, dim=0),
        torch.stack(gt_out, dim=0),
        torch.stack(m_out, dim=0),
    )


def _compute_split_counts(n_samples: int, train_ratio: float, val_ratio: float, test_ratio: float):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")
    if n_samples < 3:
        raise RuntimeError("Need at least 3 samples for train/val/test splits.")

    n_train = max(1, int(n_samples * train_ratio))
    n_val = max(1, int(n_samples * val_ratio))
    n_test = n_samples - n_train - n_val
    while n_test < 1:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise RuntimeError("Unable to build non-empty train/val/test splits.")
        n_test = n_samples - n_train - n_val
    return n_train, n_val, n_test


def make_loaders(
    vis_dir: str,
    ir_dir: str,
    gt_dir: str,
    size: int,
    batch_size: int,
    workers: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    train_aug: bool,
    native_res_train: bool,
    pad_multiple: int,
    max_train_side: int,
    vis_brightness_jitter: float,
    vis_contrast_jitter: float,
    ir_brightness_jitter: float,
    ir_contrast_jitter: float,
    gt_brightness_jitter: float,
    gt_contrast_jitter: float,
):
    samples = match_triplets(vis_dir, ir_dir, gt_dir)
    n_train, n_val, n_test = _compute_split_counts(len(samples), train_ratio, val_ratio, test_ratio)
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(samples), generator=gen).tolist()
    tr_ids = order[:n_train]
    vl_ids = order[n_train : n_train + n_val]
    te_ids = order[n_train + n_val :]

    tr_ds = FusionDataset(
        samples,
        size=size,
        augment=train_aug,
        native_res=native_res_train,
        max_side=max_train_side,
        vis_brightness_jitter=vis_brightness_jitter,
        vis_contrast_jitter=vis_contrast_jitter,
        ir_brightness_jitter=ir_brightness_jitter,
        ir_contrast_jitter=ir_contrast_jitter,
        gt_brightness_jitter=gt_brightness_jitter,
        gt_contrast_jitter=gt_contrast_jitter,
        indices=tr_ids,
    )
    vl_ds = FusionDataset(samples, size=size, augment=False, indices=vl_ids)
    te_ds = FusionDataset(samples, size=size, augment=False, indices=te_ids)
    tr_collate = partial(native_pad_collate, pad_multiple=pad_multiple) if native_res_train else None

    pin = torch.cuda.is_available()
    tr_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
        collate_fn=tr_collate,
    )
    vl_loader = DataLoader(
        vl_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
    )
    te_loader = DataLoader(
        te_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
    )
    return tr_loader, vl_loader, te_loader, len(samples), n_train, n_val, n_test


# =========================================================
# MODEL
# =========================================================
class CrossSRA(nn.Module):
    def __init__(self, dim: int, heads: int = 8, sr: int = 4):
        super().__init__()
        self.h = heads
        self.scale = (dim // heads) ** -0.5
        self.sr = sr
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        if sr > 1:
            self.conv = nn.Conv2d(dim, dim, sr, sr)
            self.norm = nn.LayerNorm(dim)

    def forward(self, qx, kvx, h, w, kv_bias=None):
        bsz, n, c = qx.shape
        q = self.q(qx).reshape(bsz, n, self.h, c // self.h).permute(0, 2, 1, 3)

        kv = kvx.permute(0, 2, 1).reshape(bsz, c, h, w)
        kv_h, kv_w = h, w
        if self.sr > 1:
            kv = self.conv(kv)
            kv_h, kv_w = kv.shape[-2], kv.shape[-1]
            kv = kv.flatten(2).transpose(1, 2)
            kv = self.norm(kv)
        else:
            kv = kv.flatten(2).transpose(1, 2)

        k = self.k(kv).reshape(bsz, -1, self.h, c // self.h).permute(0, 2, 1, 3)
        v = self.v(kv).reshape(bsz, -1, self.h, c // self.h).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if kv_bias is not None:
            if kv_bias.ndim == 4:
                kv_bias = F.interpolate(kv_bias, size=(kv_h, kv_w), mode="bilinear", align_corners=False)
                kv_bias = kv_bias.flatten(1)
            attn = attn + kv_bias[:, None, None, :].to(dtype=attn.dtype)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, n, c)
        return self.proj(x)


class FusionTokenBlock(nn.Module):
    def __init__(self, dim: int, heads: int, sr: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.v2i = CrossSRA(dim, heads=heads, sr=sr)
        self.i2v = CrossSRA(dim, heads=heads, sr=sr)
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, vf, if_, h, w, vis_bias=None, ir_bias=None, attn_bias_gamma: float = 0.0):
        kv_bias_ir = None if ir_bias is None else (attn_bias_gamma * ir_bias)
        kv_bias_vis = None if vis_bias is None else (attn_bias_gamma * vis_bias)
        attn = torch.sigmoid(self.a) * self.v2i(vf, if_, h, w, kv_bias=kv_bias_ir)
        attn = attn + torch.sigmoid(self.b) * self.i2v(if_, vf, h, w, kv_bias=kv_bias_vis)
        vf = vf + attn
        vf = vf + self.ffn(self.norm(vf))
        return vf


def down(ic, oc):
    return nn.Sequential(nn.Conv2d(ic, oc, 3, 2, 1), nn.BatchNorm2d(oc), nn.LeakyReLU(0.2, inplace=True))


def up(ic, oc):
    return nn.Sequential(
        nn.ConvTranspose2d(ic, oc, 2, 2),
        nn.BatchNorm2d(oc),
        nn.LeakyReLU(0.2, inplace=True),
    )


def conv(ic, oc):
    return nn.Sequential(nn.Conv2d(ic, oc, 3, 1, 1), nn.BatchNorm2d(oc), nn.LeakyReLU(0.2, inplace=True))


class FusionNet(nn.Module):
    """
    KPFuse v2:
    - 4-layer encoder/decoder (reduced from 5)
    - 3 outputs:
      1) VIS-dominant fusion
      2) IR-dominant fusion
      3) exclusivity-weighted VIS/IR fusion
    """

    def __init__(
        self,
        base_ch: int = 32,
        bottleneck_ch: int = 256,
        attn_heads: int = 8,
        attn_depth_l3: int = 1,
        attn_depth_l4: int = 1,
        attn_sr_l3: int = 1,
        attn_sr_l4: int = 2,
        attn_mlp_ratio: float = 2.0,
        gate_alpha_vis: float = 0.10,
        gate_alpha_ir: float = 0.15,
        attn_bias_gamma: float = 0.10,
        attn_bias_detach: bool = True,
        luma_pred_mix: float = 0.12,
        pred_rgb_mix: float = 0.10,
        excl_smooth_mode: str = "gaussian",
        excl_smooth_kernel: int = 7,
        excl_smooth_sigma: float = 1.6,
        excl_smooth_passes: int = 1,
        kmap_smooth_mode: str = "gaussian",
        kmap_smooth_kernel: int = 3,
        kmap_smooth_sigma: float = 0.9,
        kmap_smooth_passes: int = 1,
        gain_eps: float = 1e-3,
        gain_min: float = 0.4,
        gain_max: float = 2.2,
        gain_smooth_mode: str = "gaussian",
        gain_smooth_kernel: int = 5,
        gain_smooth_sigma: float = 1.1,
        gain_smooth_passes: int = 1,
        gain_strength: float = 0.7,
    ):
        super().__init__()
        ch1 = base_ch
        ch2 = base_ch * 2
        ch3 = base_ch * 4
        ch4 = bottleneck_ch
        if (attn_depth_l3 > 0) and (ch3 % attn_heads != 0):
            raise ValueError("Layer-3 channels must be divisible by attn_heads when attn_depth_l3>0")
        if (attn_depth_l4 > 0) and (ch4 % attn_heads != 0):
            raise ValueError("Layer-4 channels must be divisible by attn_heads when attn_depth_l4>0")

        # 4-layer encoder
        self.v1 = down(3, ch1)
        self.v2 = down(ch1, ch2)
        self.v3 = down(ch2, ch3)
        self.v4 = down(ch3, ch4)
        self.i1 = down(1, ch1)
        self.i2 = down(ch1, ch2)
        self.i3 = down(ch2, ch3)
        self.i4 = down(ch3, ch4)

        # Multi-scale fusion stages (L3 + L4/bottleneck)
        self.fusion_blocks_l3 = nn.ModuleList(
            [
                FusionTokenBlock(
                    dim=ch3,
                    heads=attn_heads,
                    sr=attn_sr_l3,
                    mlp_ratio=attn_mlp_ratio,
                )
                for _ in range(attn_depth_l3)
            ]
        )
        self.fusion_blocks_l4 = nn.ModuleList(
            [
                FusionTokenBlock(
                    dim=ch4,
                    heads=attn_heads,
                    sr=attn_sr_l4,
                    mlp_ratio=attn_mlp_ratio,
                )
                for _ in range(attn_depth_l4)
            ]
        )

        # 4-layer decoder
        self.up4 = up(ch4, ch3)
        self.c4 = conv(ch3 + ch3 + ch3, ch3)
        self.up3 = up(ch3, ch2)
        self.c3 = conv(ch2 + ch2 + ch2, ch2)
        self.up2 = up(ch2, ch1)
        self.c2 = conv(ch1 + ch1 + ch1, ch1)
        self.up1 = up(ch1, ch1)
        self.out = nn.Conv2d(ch1, 3, 3, 1, 1)

        self.gate_alpha_vis = gate_alpha_vis
        self.gate_alpha_ir = gate_alpha_ir
        self.attn_bias_gamma = attn_bias_gamma
        self.attn_bias_detach = attn_bias_detach
        self.luma_pred_mix = luma_pred_mix
        self.pred_rgb_mix = pred_rgb_mix
        self.excl_smooth_mode = excl_smooth_mode
        self.excl_smooth_kernel = excl_smooth_kernel
        self.excl_smooth_sigma = excl_smooth_sigma
        self.excl_smooth_passes = excl_smooth_passes
        self.kmap_smooth_mode = kmap_smooth_mode
        self.kmap_smooth_kernel = kmap_smooth_kernel
        self.kmap_smooth_sigma = kmap_smooth_sigma
        self.kmap_smooth_passes = kmap_smooth_passes
        self.gain_eps = max(1e-6, float(gain_eps))
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        if self.gain_max < self.gain_min:
            raise ValueError("gain_max must be >= gain_min")
        self.gain_smooth_mode = gain_smooth_mode
        self.gain_smooth_kernel = gain_smooth_kernel
        self.gain_smooth_sigma = gain_smooth_sigma
        self.gain_smooth_passes = gain_smooth_passes
        self.gain_strength = min(1.0, max(0.0, float(gain_strength)))

    def _run_fusion_stage(
        self,
        vis_feat: torch.Tensor,
        ir_feat: torch.Tensor,
        blocks: nn.ModuleList,
        vis_kmap: Optional[torch.Tensor],
        ir_kmap: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if len(blocks) == 0:
            return vis_feat

        vis_bias = None
        if vis_kmap is not None:
            vis_bias = F.interpolate(vis_kmap, size=vis_feat.shape[-2:], mode="bilinear", align_corners=False)
            vis_bias = smooth_spatial_map(
                vis_bias,
                kernel_size=self.kmap_smooth_kernel,
                sigma=self.kmap_smooth_sigma,
                passes=self.kmap_smooth_passes,
                mode=self.kmap_smooth_mode,
            )
            if self.attn_bias_detach:
                vis_bias = vis_bias.detach()
            vis_feat = vis_feat * (1.0 + self.gate_alpha_vis * vis_bias)

        ir_bias = None
        if ir_kmap is not None:
            ir_bias = F.interpolate(ir_kmap, size=ir_feat.shape[-2:], mode="bilinear", align_corners=False)
            ir_bias = smooth_spatial_map(
                ir_bias,
                kernel_size=self.kmap_smooth_kernel,
                sigma=self.kmap_smooth_sigma,
                passes=self.kmap_smooth_passes,
                mode=self.kmap_smooth_mode,
            )
            if self.attn_bias_detach:
                ir_bias = ir_bias.detach()
            ir_feat = ir_feat * (1.0 + self.gate_alpha_ir * ir_bias)

        bsz, c, h, w = vis_feat.shape
        vis_tokens = vis_feat.flatten(2).transpose(1, 2)
        ir_tokens = ir_feat.flatten(2).transpose(1, 2)
        for block in blocks:
            vis_tokens = block(
                vis_tokens,
                ir_tokens,
                h,
                w,
                vis_bias=vis_bias,
                ir_bias=ir_bias,
                attn_bias_gamma=self.attn_bias_gamma,
            )
        return vis_tokens.transpose(1, 2).reshape(bsz, c, h, w)

    def _compute_exclusive_ir_weight(
        self,
        v: torch.Tensor,
        i: torch.Tensor,
        vis_kmap: Optional[torch.Tensor],
        ir_kmap: Optional[torch.Tensor],
    ) -> torch.Tensor:
        vis_y = rgb_to_luma(v)
        vis_grad = normalize_map_per_sample(gradient_mag_map(vis_y))
        ir_grad = normalize_map_per_sample(gradient_mag_map(i))

        if vis_kmap is None:
            vis_k = vis_grad
        else:
            vis_k = F.interpolate(vis_kmap, size=vis_y.shape[-2:], mode="bilinear", align_corners=False)
            vis_k = normalize_map_per_sample(vis_k)

        if ir_kmap is None:
            ir_k = ir_grad
        else:
            ir_k = F.interpolate(ir_kmap, size=i.shape[-2:], mode="bilinear", align_corners=False)
            ir_k = normalize_map_per_sample(ir_k)

        # Exclusive information proxies:
        # where one modality has stronger gradients/keypoint response than the other.
        vis_exclusive = F.relu(vis_grad - ir_grad) + F.relu(vis_k - ir_k)
        ir_exclusive = F.relu(ir_grad - vis_grad) + F.relu(ir_k - vis_k)
        ir_w = ir_exclusive / (vis_exclusive + ir_exclusive + 1e-6)
        ir_w = ir_w.clamp(0.0, 1.0)
        ir_w = smooth_spatial_map(
            ir_w,
            kernel_size=self.excl_smooth_kernel,
            sigma=self.excl_smooth_sigma,
            passes=self.excl_smooth_passes,
            mode=self.excl_smooth_mode,
        )
        return ir_w.clamp(0.0, 1.0)

    def _adaptive_blend(self, v, i, pred_rgb, ir_priority):
        vis_y = rgb_to_luma(v)
        pred_y = rgb_to_luma(pred_rgb)
        target_y = ((1.0 - ir_priority) * vis_y) + (ir_priority * i)
        fused_y = ((1.0 - self.luma_pred_mix) * target_y) + (self.luma_pred_mix * pred_y)
        gain = (fused_y / (vis_y + self.gain_eps)).clamp(self.gain_min, self.gain_max)
        gain = smooth_spatial_map(
            gain,
            kernel_size=self.gain_smooth_kernel,
            sigma=self.gain_smooth_sigma,
            passes=self.gain_smooth_passes,
            mode=self.gain_smooth_mode,
        )
        gain = ((1.0 - self.gain_strength) + (self.gain_strength * gain)).clamp(self.gain_min, self.gain_max)
        vis_color_preserved = (v * gain).clamp(0.0, 1.0)
        fused_rgb = ((1.0 - self.pred_rgb_mix) * vis_color_preserved) + (self.pred_rgb_mix * pred_rgb)
        return fused_rgb.clamp(0.0, 1.0)

    def forward(self, v, i, vis_kmap=None, ir_kmap=None, return_aux: bool = False, return_all: bool = False):
        v1 = self.v1(v)
        v2 = self.v2(v1)
        v3 = self.v3(v2)
        i1 = self.i1(i)
        i2 = self.i2(i1)
        i3 = self.i3(i2)

        v3_f = self._run_fusion_stage(v3, i3, self.fusion_blocks_l3, vis_kmap, ir_kmap)
        v4 = self.v4(v3_f)
        i4 = self.i4(i3)
        v4_f = self._run_fusion_stage(v4, i4, self.fusion_blocks_l4, vis_kmap, ir_kmap)

        u4 = self.up4(v4_f)
        if u4.shape[-2:] != v3_f.shape[-2:]:
            u4 = F.interpolate(u4, size=v3_f.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.c4(torch.cat([u4, v3_f, i3], dim=1))

        u3 = self.up3(d4)
        if u3.shape[-2:] != v2.shape[-2:]:
            u3 = F.interpolate(u3, size=v2.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.c3(torch.cat([u3, v2, i2], dim=1))

        u2 = self.up2(d3)
        if u2.shape[-2:] != v1.shape[-2:]:
            u2 = F.interpolate(u2, size=v1.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.c2(torch.cat([u2, v1, i1], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != v.shape[-2:]:
            d1 = F.interpolate(d1, size=v.shape[-2:], mode="bilinear", align_corners=False)
        pred_rgb = torch.sigmoid(self.out(d1))

        ir_priority_excl = self._compute_exclusive_ir_weight(v, i, vis_kmap, ir_kmap)
        ir_priority_vis = (0.25 * ir_priority_excl).clamp(0.0, 0.35)
        ir_priority_ir = (0.65 + (0.35 * ir_priority_excl)).clamp(0.65, 1.0)

        fused_vis = self._adaptive_blend(v, i, pred_rgb, ir_priority_vis)
        fused_ir = self._adaptive_blend(v, i, pred_rgb, ir_priority_ir)
        fused_excl = self._adaptive_blend(v, i, pred_rgb, ir_priority_excl)

        if return_all:
            return fused_vis, fused_ir, fused_excl
        if return_aux:
            return fused_excl, {
                "fused_vis": fused_vis,
                "fused_ir": fused_ir,
                "fused_excl": fused_excl,
                "ir_priority_vis": ir_priority_vis,
                "ir_priority_ir": ir_priority_ir,
                "ir_priority_excl": ir_priority_excl,
                "pred_rgb": pred_rgb,
            }
        return fused_excl


# =========================================================
# KEYNET LOSSES
# =========================================================
class KeyNetResponse(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.keynet = KF.KeyNet(pretrained=True)
        except Exception:
            self.keynet = KF.KeyNet(pretrained=False)
        self.keynet.eval()
        for p in self.keynet.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.keynet(x)
        if isinstance(out, dict):
            for key in ("response", "scores", "score_map"):
                if key in out:
                    out = out[key]
                    break
            else:
                raise RuntimeError("Unexpected KeyNet dict output format.")
        if out.ndim == 3:
            out = out.unsqueeze(1)
        if out.ndim != 4:
            raise RuntimeError(f"Unexpected KeyNet output shape: {out.shape}")
        return out


class KeyNetLoss(nn.Module):
    def __init__(self, vis_weight: float = 1.0, ir_weight: float = 2.0, detector_max_side: int = 256):
        super().__init__()
        self.detector = KeyNetResponse()
        self.w_vis = vis_weight
        self.w_ir = ir_weight
        self.detector_max_side = int(detector_max_side)

    @staticmethod
    def _normalize_map(x: torch.Tensor) -> torch.Tensor:
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-6)

    @staticmethod
    def _masked_l1(a: torch.Tensor, b: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        diff = (a - b).abs()
        if valid_mask is None:
            return diff.mean()
        if valid_mask.shape[-2:] != diff.shape[-2:]:
            valid_mask = F.interpolate(valid_mask, size=diff.shape[-2:], mode="nearest")
        valid_mask = valid_mask.to(dtype=diff.dtype)
        denom = valid_mask.sum().clamp_min(1.0)
        return (diff * valid_mask).sum() / denom

    def _resize_for_detector(self, x: torch.Tensor) -> torch.Tensor:
        if self.detector_max_side <= 0:
            return x
        h, w = x.shape[-2:]
        long_side = max(h, w)
        if long_side <= self.detector_max_side:
            return x
        scale = self.detector_max_side / float(long_side)
        new_h = max(8, int(round(h * scale)))
        new_w = max(8, int(round(w * scale)))
        return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    def forward(self, v, i, f, valid_mask: Optional[torch.Tensor] = None):
        detector_dtype = next(self.detector.parameters()).dtype
        v = v.mean(1, True).to(dtype=detector_dtype)
        i = i.to(dtype=detector_dtype)
        f = f.mean(1, True).to(dtype=detector_dtype)
        v = self._resize_for_detector(v)
        i = self._resize_for_detector(i)
        f = self._resize_for_detector(f)
        with torch.no_grad():
            rv = self.detector(v)
            ri = self.detector(i)
        rf = self.detector(f)
        if rf.shape[-2:] != rv.shape[-2:]:
            rf = F.interpolate(rf, size=rv.shape[-2:], mode="bilinear", align_corners=False)
        rv = self._normalize_map(rv)
        ri = self._normalize_map(ri)
        rf = self._normalize_map(rf)
        vis_term = self._masked_l1(rf, rv, valid_mask)
        ir_term = self._masked_l1(rf, ri, valid_mask)
        return (self.w_vis * vis_term) + (self.w_ir * ir_term)


class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)


class Loss(nn.Module):
    # Same objective family as kpfuse_v1, with strong keypoint-retention terms.
    def __init__(
        self,
        w_gt_l1: float = 1.0,
        w_ssim: float = 0.9,
        w_grad: float = 0.7,
        w_src_l1: float = 0.25,
        w_sp: float = 0.08,
        sp_vis_weight: float = 1.0,
        sp_ir_weight: float = 3.0,
        gt_anchor: float = 0.02,
        w_vis_luma: float = 1.25,
        w_ir_luma: float = 0.80,
        w_vis_contrast: float = 0.85,
        w_union: float = 0.6,
        sp_detector_max_side: int = 256,
    ):
        super().__init__()
        self.sp = KeyNetLoss(
            vis_weight=sp_vis_weight,
            ir_weight=sp_ir_weight,
            detector_max_side=sp_detector_max_side,
        )
        self.sobel = SobelGrad()
        self.w_gt_l1 = w_gt_l1
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.w_src_l1 = w_src_l1
        self.w_sp = w_sp
        self.gt_anchor = gt_anchor
        self.w_vis_luma = w_vis_luma
        self.w_ir_luma = w_ir_luma
        self.w_vis_contrast = w_vis_contrast
        self.w_union = w_union

    @staticmethod
    def _masked_mean(x: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if valid_mask is None:
            return x.mean()
        if valid_mask.shape[-2:] != x.shape[-2:]:
            valid_mask = F.interpolate(valid_mask, size=x.shape[-2:], mode="nearest")
        valid_mask = valid_mask.to(dtype=x.dtype)
        denom = valid_mask.sum().clamp_min(1.0)
        return (x * valid_mask).sum() / denom

    @staticmethod
    def _union_keypoint_l1(
        v: torch.Tensor, i: torch.Tensor, f: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        vis_y = rgb_to_luma(v)
        fused_y = rgb_to_luma(f)
        vis_grad = normalize_map_per_sample(gradient_mag_map(vis_y))
        ir_grad = normalize_map_per_sample(gradient_mag_map(i))
        fused_grad = normalize_map_per_sample(gradient_mag_map(fused_y))
        union_src = torch.maximum(vis_grad, ir_grad)
        return Loss._masked_mean((fused_grad - union_src).abs(), valid_mask)

    def forward(
        self,
        v,
        i,
        gt,
        f,
        ir_priority=None,
        valid_mask: Optional[torch.Tensor] = None,
        sp_branch_scale: float = 1.0,
    ):
        if ir_priority is None:
            ir_priority = torch.full_like(i, 0.5)
        vis_priority = 1.0 - ir_priority

        src_target = torch.max(v, i.repeat(1, 3, 1, 1))
        gt_l1 = self._masked_mean((f - gt).abs(), valid_mask)
        src_l1 = self._masked_mean((f - src_target).abs(), valid_mask)

        ssim_term = K.losses.ssim_loss(f, gt, window_size=11)
        if ssim_term.ndim >= 4:
            ssim_term = self._masked_mean(ssim_term, valid_mask)
        elif ssim_term.ndim > 0:
            ssim_term = ssim_term.mean()

        grad_term = self._masked_mean(
            (self.sobel(f.mean(1, True)) - self.sobel(gt.mean(1, True))).abs(),
            valid_mask,
        )
        gt_term = (self.w_gt_l1 * gt_l1) + (self.w_ssim * ssim_term) + (self.w_grad * grad_term)

        fused_y = rgb_to_luma(f)
        vis_y = rgb_to_luma(v)
        vis_luma_term = self._masked_mean(vis_priority * (fused_y - vis_y).abs(), valid_mask)
        ir_luma_term = self._masked_mean(ir_priority * (fused_y - i).abs(), valid_mask)

        fused_contrast = local_contrast_map(fused_y, kernel_size=5)
        vis_contrast = local_contrast_map(vis_y, kernel_size=5)
        vis_contrast_term = self._masked_mean(
            vis_priority * (fused_contrast - vis_contrast).abs(),
            valid_mask,
        )
        union_term = self._union_keypoint_l1(v, i, f, valid_mask=valid_mask)
        if (self.w_sp > 0.0) and (sp_branch_scale > 0.0):
            sp_term = self.sp(v, i, f, valid_mask=valid_mask)
        else:
            sp_term = torch.zeros((), device=f.device, dtype=f.dtype)

        return (
            self.gt_anchor * gt_term
            + self.w_src_l1 * src_l1
            + self.w_vis_luma * vis_luma_term
            + self.w_ir_luma * ir_luma_term
            + self.w_vis_contrast * vis_contrast_term
            + self.w_union * union_term
            + (self.w_sp * sp_branch_scale * sp_term)
        )


# =========================================================
# EVAL + TRAIN
# =========================================================
@torch.no_grad()
def compute_keypoint_maps(v: torch.Tensor, i: torch.Tensor, detector: nn.Module):
    detector.eval()
    detector_dtype = next(detector.parameters()).dtype
    vg = v.mean(1, True).to(dtype=detector_dtype)
    ig = i.to(dtype=detector_dtype)
    rv = normalize_map_per_sample(detector(vg))
    ri = normalize_map_per_sample(detector(ig))
    return rv, ri


@torch.no_grad()
def evaluate_keypoint_retention(net, detector, loader, device, use_amp, topk: int):
    autocast_dev = "cuda" if device.type == "cuda" else "cpu"
    net.eval()
    detector.eval()
    detector_dtype = next(detector.parameters()).dtype
    branch_acc = {
        "VISdom": {"vis": 0.0, "ir": 0.0},
        "IRdom": {"vis": 0.0, "ir": 0.0},
        "EXCL": {"vis": 0.0, "ir": 0.0},
    }

    def _branch_retention(rv: torch.Tensor, ri: torch.Tensor, rf: torch.Tensor):
        rvf = rv.flatten(1)
        rif = ri.flatten(1)
        rff = rf.flatten(1)
        k = min(topk, rvf.shape[1], rif.shape[1], rff.shape[1])
        vis_idx = rvf.topk(k, dim=1).indices
        ir_idx = rif.topk(k, dim=1).indices
        fused_idx = rff.topk(k, dim=1).indices
        vis_mask = torch.zeros_like(rvf)
        ir_mask = torch.zeros_like(rif)
        fused_mask = torch.zeros_like(rff)
        vis_mask.scatter_(1, vis_idx, 1.0)
        ir_mask.scatter_(1, ir_idx, 1.0)
        fused_mask.scatter_(1, fused_idx, 1.0)
        vis_ret = (vis_mask * fused_mask).sum(dim=1) / vis_mask.sum(dim=1).clamp_min(1.0)
        ir_ret = (ir_mask * fused_mask).sum(dim=1) / ir_mask.sum(dim=1).clamp_min(1.0)
        return vis_ret, ir_ret

    n = 0
    for batch in loader:
        if len(batch) == 4:
            v, i, _gt, _mask = batch
        else:
            v, i, _gt = batch
        v = v.to(device, non_blocking=True)
        i = i.to(device, non_blocking=True)
        with torch.amp.autocast(autocast_dev, enabled=use_amp):
            vis_kmap, ir_kmap = compute_keypoint_maps(v, i, detector)
            _, aux = net(v, i, vis_kmap=vis_kmap, ir_kmap=ir_kmap, return_aux=True)
        rv = vis_kmap.to(dtype=detector_dtype)
        ri = ir_kmap.to(dtype=detector_dtype)

        for branch_name, fused_key in (
            ("VISdom", "fused_vis"),
            ("IRdom", "fused_ir"),
            ("EXCL", "fused_excl"),
        ):
            fg = aux[fused_key].mean(1, True).to(dtype=detector_dtype)
            rf = normalize_map_per_sample(detector(fg))
            vis_ret, ir_ret = _branch_retention(rv, ri, rf)
            branch_acc[branch_name]["vis"] += vis_ret.sum().item()
            branch_acc[branch_name]["ir"] += ir_ret.sum().item()
        n += v.shape[0]

    if n == 0:
        return {
            "VISdom": (0.0, 0.0, 0.0),
            "IRdom": (0.0, 0.0, 0.0),
            "EXCL": (0.0, 0.0, 0.0),
        }

    out = {}
    for branch_name in ("VISdom", "IRdom", "EXCL"):
        vis_score = branch_acc[branch_name]["vis"] / n
        ir_score = branch_acc[branch_name]["ir"] / n
        out[branch_name] = (vis_score, ir_score, 0.5 * (vis_score + ir_score))
    return out


def _resolve_view_weights(n_views: int, raw_weights: Sequence[float]) -> List[float]:
    if n_views < 1:
        return [1.0]
    ww = [max(0.0, float(x)) for x in raw_weights[:n_views]]
    if len(ww) < n_views:
        ww.extend([ww[-1] if ww else 1.0] * (n_views - len(ww)))
    s = sum(ww)
    if s <= 1e-8:
        return [1.0 / n_views] * n_views
    return [w / s for w in ww]


def _predict_with_multi_view(
    net,
    detector,
    v,
    i,
    args,
    autocast_dev: str,
    use_amp: bool,
):
    views = max(1, args.train_views)
    view_weights = _resolve_view_weights(views, args.view_loss_weights)
    acc = {
        "fused_vis": [],
        "fused_ir": [],
        "fused_excl": [],
        "ir_priority_vis": [],
        "ir_priority_ir": [],
        "ir_priority_excl": [],
    }
    for vi in range(views):
        if vi == 0:
            vv, ii = v, i
        else:
            vv = apply_tensor_photometric(v, args.vis_brightness_jitter, args.vis_contrast_jitter)
            ii = apply_tensor_photometric(i, args.ir_brightness_jitter, args.ir_contrast_jitter)
            if args.view_prefix_mix > 0:
                mix = min(1.0, max(0.0, args.view_prefix_mix))
                vv = ((1.0 - mix) * vv) + (mix * v)
                ii = ((1.0 - mix) * ii) + (mix * i)

        with torch.amp.autocast(autocast_dev, enabled=use_amp):
            vis_kmap, ir_kmap = compute_keypoint_maps(vv, ii, detector)
            _, aux = net(vv, ii, vis_kmap=vis_kmap, ir_kmap=ir_kmap, return_aux=True)
        for k in acc:
            acc[k].append(aux[k])

    out = {}
    for k in acc:
        stacked = torch.stack(acc[k], dim=0)
        w = torch.tensor(view_weights, device=stacked.device, dtype=stacked.dtype).view(-1, 1, 1, 1, 1)
        out[k] = (stacked * w).sum(dim=0)
    return out


@torch.no_grad()
def save_epoch_previews(net, detector, loader, device, use_amp, out_root, epoch: int, max_items: int = 6):
    if max_items <= 0:
        return
    out_dir = Path(out_root) / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    autocast_dev = "cuda" if device.type == "cuda" else "cpu"
    net.eval()
    saved = 0
    for batch in loader:
        if len(batch) == 4:
            v, i, gt, _mask = batch
        else:
            v, i, gt = batch
        v = v.to(device, non_blocking=True)
        i = i.to(device, non_blocking=True)
        vis_kmap, ir_kmap = compute_keypoint_maps(v, i, detector)
        with torch.amp.autocast(autocast_dev, enabled=use_amp):
            _, aux = net(v, i, vis_kmap=vis_kmap, ir_kmap=ir_kmap, return_aux=True)
        v = v.cpu()
        i = i.cpu()
        gt = gt.cpu()
        f_vis = aux["fused_vis"].detach().cpu().clamp_(0.0, 1.0)
        f_ir = aux["fused_ir"].detach().cpu().clamp_(0.0, 1.0)
        f_excl = aux["fused_excl"].detach().cpu().clamp_(0.0, 1.0)
        for bi in range(v.shape[0]):
            stem = f"sample_{saved + 1:04d}"
            vutils.save_image(f_vis[bi], str(out_dir / f"{stem}_vis_dom.png"))
            vutils.save_image(f_ir[bi], str(out_dir / f"{stem}_ir_dom.png"))
            vutils.save_image(f_excl[bi], str(out_dir / f"{stem}_exclusive.png"))
            ir_rgb = i[bi].repeat(3, 1, 1)
            panel = torch.cat([v[bi], ir_rgb, f_vis[bi], f_ir[bi], f_excl[bi], gt[bi]], dim=2)
            vutils.save_image(panel, str(out_dir / f"{stem}_panel.png"))
            saved += 1
            if saved >= max_items:
                return


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    autocast_dev = "cuda" if use_amp else "cpu"
    run_start = time.perf_counter()

    tr_loader, vl_loader, te_loader, n_samples, n_train, n_val, n_test = make_loaders(
        args.vis_dir,
        args.ir_dir,
        args.gt_dir,
        args.size,
        args.batch_size,
        args.workers,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.train_aug,
        args.native_res_train,
        args.pad_multiple,
        args.max_train_side,
        args.vis_brightness_jitter,
        args.vis_contrast_jitter,
        args.ir_brightness_jitter,
        args.ir_contrast_jitter,
        args.gt_brightness_jitter,
        args.gt_contrast_jitter,
    )
    if args.native_res_train:
        print(
            "Native-resolution train mode: enabled | "
            f"pad_multiple={args.pad_multiple} max_train_side={args.max_train_side}"
        )
    print(
        f"Matched samples: {n_samples} | train: {n_train} ({len(tr_loader)} batches) | "
        f"val: {n_val} ({len(vl_loader)} batches) | test: {n_test} ({len(te_loader)} batches)"
    )

    net = FusionNet(
        base_ch=args.base_ch,
        bottleneck_ch=args.bottleneck_ch,
        attn_heads=args.attn_heads,
        attn_depth_l3=args.attn_depth_l3,
        attn_depth_l4=args.attn_depth_l4,
        attn_sr_l3=args.attn_sr_l3,
        attn_sr_l4=args.attn_sr_l4,
        attn_mlp_ratio=args.attn_mlp_ratio,
        gate_alpha_vis=args.gate_alpha_vis,
        gate_alpha_ir=args.gate_alpha_ir,
        attn_bias_gamma=args.attn_bias_gamma,
        attn_bias_detach=args.attn_bias_detach,
        luma_pred_mix=args.luma_pred_mix,
        pred_rgb_mix=args.pred_rgb_mix,
        excl_smooth_mode=args.excl_smooth_mode,
        excl_smooth_kernel=args.excl_smooth_kernel,
        excl_smooth_sigma=args.excl_smooth_sigma,
        excl_smooth_passes=args.excl_smooth_passes,
        kmap_smooth_mode=args.kmap_smooth_mode,
        kmap_smooth_kernel=args.kmap_smooth_kernel,
        kmap_smooth_sigma=args.kmap_smooth_sigma,
        kmap_smooth_passes=args.kmap_smooth_passes,
        gain_eps=args.gain_eps,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        gain_smooth_mode=args.gain_smooth_mode,
        gain_smooth_kernel=args.gain_smooth_kernel,
        gain_smooth_sigma=args.gain_smooth_sigma,
        gain_smooth_passes=args.gain_smooth_passes,
        gain_strength=args.gain_strength,
    ).to(device)

    loss_fn = Loss(
        w_gt_l1=args.w_gt_l1,
        w_ssim=args.w_ssim,
        w_grad=args.w_grad,
        w_src_l1=args.w_src_l1,
        w_sp=args.w_sp,
        sp_vis_weight=args.sp_vis_weight,
        sp_ir_weight=args.sp_ir_weight,
        gt_anchor=args.gt_anchor,
        w_vis_luma=args.w_vis_luma,
        w_ir_luma=args.w_ir_luma,
        w_vis_contrast=args.w_vis_contrast,
        w_union=args.w_sp_union,
        sp_detector_max_side=args.sp_detector_max_side,
    ).to(device)
    loss_fn.sp.eval()

    w_sum = args.loss_w_vis + args.loss_w_ir + args.loss_w_excl
    lw_vis = args.loss_w_vis / w_sum
    lw_ir = args.loss_w_ir / w_sum
    lw_excl = args.loss_w_excl / w_sum
    print(f"Loss mixing: vis={lw_vis:.3f} ir={lw_ir:.3f} excl={lw_excl:.3f}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=args.lr_decay,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    os.makedirs(args.preview_dir, exist_ok=True)

    best = float("inf")
    stall = 0
    for e in range(args.epochs):
        ep_start = time.perf_counter()
        net.train()
        tl = 0.0
        for batch in tr_loader:
            if len(batch) == 4:
                v, i, gt, valid_mask = batch
            else:
                v, i, gt = batch
                valid_mask = None
            v = v.to(device, non_blocking=True)
            i = i.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            if valid_mask is not None:
                valid_mask = valid_mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            preds = _predict_with_multi_view(net, loss_fn.sp.detector, v, i, args, autocast_dev, use_amp)
            loss_vis = loss_fn(
                v,
                i,
                gt,
                preds["fused_vis"],
                ir_priority=preds["ir_priority_vis"],
                valid_mask=valid_mask,
                sp_branch_scale=args.sp_scale_vis,
            )
            loss_ir = loss_fn(
                v,
                i,
                gt,
                preds["fused_ir"],
                ir_priority=preds["ir_priority_ir"],
                valid_mask=valid_mask,
                sp_branch_scale=args.sp_scale_ir,
            )
            loss_excl = loss_fn(
                v,
                i,
                gt,
                preds["fused_excl"],
                ir_priority=preds["ir_priority_excl"],
                valid_mask=valid_mask,
                sp_branch_scale=args.sp_scale_excl,
            )
            loss = (lw_vis * loss_vis) + (lw_ir * loss_ir) + (lw_excl * loss_excl)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            tl += loss.item()

        net.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in vl_loader:
                if len(batch) == 4:
                    v, i, gt, valid_mask = batch
                else:
                    v, i, gt = batch
                    valid_mask = None
                v = v.to(device, non_blocking=True)
                i = i.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                if valid_mask is not None:
                    valid_mask = valid_mask.to(device, non_blocking=True)
                preds = _predict_with_multi_view(net, loss_fn.sp.detector, v, i, args, autocast_dev, use_amp)
                lv = loss_fn(
                    v,
                    i,
                    gt,
                    preds["fused_vis"],
                    ir_priority=preds["ir_priority_vis"],
                    valid_mask=valid_mask,
                    sp_branch_scale=args.sp_scale_vis,
                )
                li = loss_fn(
                    v,
                    i,
                    gt,
                    preds["fused_ir"],
                    ir_priority=preds["ir_priority_ir"],
                    valid_mask=valid_mask,
                    sp_branch_scale=args.sp_scale_ir,
                )
                le = loss_fn(
                    v,
                    i,
                    gt,
                    preds["fused_excl"],
                    ir_priority=preds["ir_priority_excl"],
                    valid_mask=valid_mask,
                    sp_branch_scale=args.sp_scale_excl,
                )
                vl += (lw_vis * lv + lw_ir * li + lw_excl * le).item()

        tl /= max(1, len(tr_loader))
        vl /= max(1, len(vl_loader))
        scheduler.step(vl)
        ret_stats = evaluate_keypoint_retention(
            net=net,
            detector=loss_fn.sp.detector,
            loader=te_loader if args.retention_split == "test" else vl_loader,
            device=device,
            use_amp=use_amp,
            topk=args.retention_topk,
        )
        vis_ret, ir_ret, mean_ret = ret_stats["EXCL"]
        print(
            f"Epoch {e + 1}/{args.epochs} | TL={tl:.4f} VL={vl:.4f} "
            f"| Ret_EXCL(vis={vis_ret:.4f}, ir={ir_ret:.4f}, mean={mean_ret:.4f}) "
            f"| LR={opt.param_groups[0]['lr']:.2e}"
        )
        r_visdom = ret_stats["VISdom"]
        r_irdom = ret_stats["IRdom"]
        print(
            f"Ret_VISdom(vis->fused={r_visdom[0]:.4f}, ir->fused={r_visdom[1]:.4f}, mean={r_visdom[2]:.4f})"
        )
        print(
            f"Ret_IRdom(vis->fused={r_irdom[0]:.4f}, ir->fused={r_irdom[1]:.4f}, mean={r_irdom[2]:.4f})"
        )
        print(
            f"Ret_EXCL(vis->fused={vis_ret:.4f}, ir->fused={ir_ret:.4f}, mean={mean_ret:.4f})"
        )

        if vl < best:
            best = vl
            stall = 0
            torch.save(net.state_dict(), args.ckpt)
            print(f"Saved best model -> {args.ckpt}")
        else:
            stall += 1

        save_epoch_previews(
            net=net,
            detector=loss_fn.sp.detector,
            loader=te_loader,
            device=device,
            use_amp=use_amp,
            out_root=args.preview_dir,
            epoch=e + 1,
            max_items=args.preview_count,
        )

        ep_sec = time.perf_counter() - ep_start
        total_min = (time.perf_counter() - run_start) / 60.0
        print(f"Epoch time: {ep_sec:.1f}s | elapsed: {total_min:.2f} min")
        if stall >= args.patience:
            print("Early stopping.")
            break


def parse_args():
    p = argparse.ArgumentParser("KPFuse v2 trainer")
    p.add_argument("--vis-dir", type=str, required=True)
    p.add_argument("--ir-dir", type=str, required=True)
    p.add_argument("--gt-dir", type=str, required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lr-decay", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=10)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", type=str, default="best_kpfuse_v2.pth")
    p.add_argument("--preview-dir", type=str, default="kpfuse_v2_previews")
    p.add_argument("--preview-count", type=int, default=6)
    p.add_argument("--retention-split", type=str, choices=["val", "test"], default="test")
    p.add_argument("--retention-topk", type=int, default=180)

    p.add_argument("--train-views", type=int, default=2)
    p.add_argument("--view-loss-weights", nargs="*", type=float, default=[1.0, 0.7])
    p.add_argument("--view-prefix-mix", type=float, default=0.25)
    p.add_argument("--train-aug", dest="train_aug", action="store_true")
    p.add_argument("--no-train-aug", dest="train_aug", action="store_false")
    p.set_defaults(train_aug=True)
    p.add_argument("--native-res-train", dest="native_res_train", action="store_true")
    p.add_argument("--no-native-res-train", dest="native_res_train", action="store_false")
    p.set_defaults(native_res_train=False)
    # 4x downsampling path benefits from 16-aligned native resolution padding.
    p.add_argument("--pad-multiple", type=int, default=16)
    p.add_argument("--max-train-side", type=int, default=0)

    p.add_argument("--vis-brightness-jitter", type=float, default=0.20)
    p.add_argument("--vis-contrast-jitter", type=float, default=0.20)
    p.add_argument("--ir-brightness-jitter", type=float, default=0.20)
    p.add_argument("--ir-contrast-jitter", type=float, default=0.20)
    p.add_argument("--gt-brightness-jitter", type=float, default=0.10)
    p.add_argument("--gt-contrast-jitter", type=float, default=0.10)

    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--bottleneck-ch", type=int, default=256)
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--attn-depth-l3", type=int, default=1)
    p.add_argument("--attn-depth-l4", type=int, default=1)
    p.add_argument("--attn-sr-l3", type=int, default=1)
    p.add_argument("--attn-sr-l4", type=int, default=2)
    p.add_argument("--attn-mlp-ratio", type=float, default=2.0)
    p.add_argument("--gate-alpha-vis", type=float, default=0.10)
    p.add_argument("--gate-alpha-ir", type=float, default=0.15)
    p.add_argument("--attn-bias-gamma", type=float, default=0.10)
    p.add_argument("--attn-bias-detach", dest="attn_bias_detach", action="store_true")
    p.add_argument("--no-attn-bias-detach", dest="attn_bias_detach", action="store_false")
    p.set_defaults(attn_bias_detach=True)
    p.add_argument("--luma-pred-mix", type=float, default=0.12)
    p.add_argument("--pred-rgb-mix", type=float, default=0.10)
    p.add_argument("--excl-smooth-mode", type=str, choices=["gaussian", "box"], default="gaussian")
    p.add_argument("--excl-smooth-kernel", type=int, default=7)
    p.add_argument("--excl-smooth-sigma", type=float, default=1.6)
    p.add_argument("--excl-smooth-passes", type=int, default=1)
    p.add_argument("--kmap-smooth-mode", type=str, choices=["gaussian", "box"], default="gaussian")
    p.add_argument("--kmap-smooth-kernel", type=int, default=3)
    p.add_argument("--kmap-smooth-sigma", type=float, default=0.9)
    p.add_argument("--kmap-smooth-passes", type=int, default=1)
    p.add_argument("--gain-eps", type=float, default=1e-3)
    p.add_argument("--gain-min", type=float, default=0.4)
    p.add_argument("--gain-max", type=float, default=2.2)
    p.add_argument("--gain-smooth-mode", type=str, choices=["gaussian", "box"], default="gaussian")
    p.add_argument("--gain-smooth-kernel", type=int, default=5)
    p.add_argument("--gain-smooth-sigma", type=float, default=1.1)
    p.add_argument("--gain-smooth-passes", type=int, default=1)
    p.add_argument("--gain-strength", type=float, default=0.7)

    p.add_argument("--w-sp-union", type=float, default=0.6)
    p.add_argument("--w-gt-l1", type=float, default=1.0)
    p.add_argument("--w-ssim", type=float, default=0.9)
    p.add_argument("--w-grad", type=float, default=0.7)
    p.add_argument("--gt-anchor", type=float, default=0.02)
    p.add_argument("--w-src-l1", type=float, default=0.25)
    p.add_argument("--w-vis-luma", type=float, default=1.25)
    p.add_argument("--w-ir-luma", type=float, default=0.8)
    p.add_argument("--w-vis-contrast", type=float, default=0.85)
    p.add_argument("--w-sp", type=float, default=0.08)
    p.add_argument("--sp-vis-weight", type=float, default=1.0)
    p.add_argument("--sp-ir-weight", type=float, default=3.0)
    p.add_argument("--sp-detector-max-side", type=int, default=256)
    p.add_argument("--sp-scale-vis", type=float, default=0.0)
    p.add_argument("--sp-scale-ir", type=float, default=0.0)
    p.add_argument("--sp-scale-excl", type=float, default=1.0)

    p.add_argument("--loss-w-vis", type=float, default=0.25)
    p.add_argument("--loss-w-ir", type=float, default=0.25)
    p.add_argument("--loss-w-excl", type=float, default=0.50)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-8:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio must equal 1.0")
    if args.train_views < 1:
        raise ValueError("--train-views must be >= 1")
    if args.pad_multiple < 0:
        raise ValueError("--pad-multiple must be >= 0")
    if args.max_train_side < 0:
        raise ValueError("--max-train-side must be >= 0")
    if args.native_res_train and args.pad_multiple == 0:
        raise ValueError("--pad-multiple must be >= 1 when --native-res-train is enabled")
    if args.excl_smooth_kernel < 1 or args.kmap_smooth_kernel < 1 or args.gain_smooth_kernel < 1:
        raise ValueError("smooth kernels must be >= 1")
    if min(args.excl_smooth_sigma, args.kmap_smooth_sigma, args.gain_smooth_sigma) <= 0:
        raise ValueError("smooth sigmas must be > 0")
    if min(args.excl_smooth_passes, args.kmap_smooth_passes, args.gain_smooth_passes) < 0:
        raise ValueError("smooth passes must be >= 0")
    if args.gain_eps <= 0:
        raise ValueError("--gain-eps must be > 0")
    if args.gain_max < args.gain_min:
        raise ValueError("--gain-max must be >= --gain-min")
    if not (0.0 <= args.gain_strength <= 1.0):
        raise ValueError("--gain-strength must be in [0, 1]")
    if args.sp_detector_max_side < 0:
        raise ValueError("--sp-detector-max-side must be >= 0")
    if min(args.sp_scale_vis, args.sp_scale_ir, args.sp_scale_excl) < 0:
        raise ValueError("--sp-scale-vis/--sp-scale-ir/--sp-scale-excl must be >= 0")
    if args.attn_depth_l3 < 0 or args.attn_depth_l4 < 0:
        raise ValueError("--attn-depth-l3/l4 must be >= 0")
    if args.attn_sr_l3 < 1 or args.attn_sr_l4 < 1:
        raise ValueError("--attn-sr-l3/l4 must be >= 1")
    if min(args.loss_w_vis, args.loss_w_ir, args.loss_w_excl) < 0:
        raise ValueError("--loss-w-vis/--loss-w-ir/--loss-w-excl must be >= 0")
    if (args.loss_w_vis + args.loss_w_ir + args.loss_w_excl) <= 0:
        raise ValueError("At least one of --loss-w-vis/--loss-w-ir/--loss-w-excl must be > 0")
    for req in [args.vis_dir, args.ir_dir, args.gt_dir]:
        if not os.path.isdir(req):
            raise FileNotFoundError(f"Missing directory: {req}")
    train(args)
