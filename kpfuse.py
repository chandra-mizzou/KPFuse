import argparse
import os
import random
import time
from pathlib import Path
from typing import Optional, Sequence
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import torchvision.utils as vutils
from torchvision.transforms import functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import kornia as K
    import kornia.feature as KF
except ImportError as exc:
    raise ImportError(
        "kpfuse.py requires kornia. Install it with: pip install kornia"
    ) from exc


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


def dark_confidence_map(vis_y: torch.Tensor, dark_thresh: float, dark_gain: float) -> torch.Tensor:
    return torch.sigmoid((dark_thresh - vis_y) * dark_gain)


def saturation_confidence_map(
    vis_rgb: torch.Tensor, sat_thresh: float, sat_gain: float
) -> torch.Tensor:
    # Near-white clipping proxy: all RGB channels are high (close to 1.0).
    min_rgb = vis_rgb.amin(dim=1, keepdim=True)
    return torch.sigmoid((min_rgb - sat_thresh) * sat_gain)


# =========================================================
# DATASET
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
        indices: Optional[Sequence[int]] = None,
    ):
        if indices is None:
            self.samples = list(samples)
        else:
            self.samples = [samples[j] for j in indices]

        if len(self.samples) < 1:
            raise RuntimeError("Dataset split has no samples.")

        self.augment = augment
        self.resize = T.Resize((size, size))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        vp, ip, gp = self.samples[i]
        with Image.open(vp) as v_img, Image.open(ip) as i_img, Image.open(gp) as g_img:
            v_img = self.resize(v_img.convert("RGB"))
            i_img = self.resize(i_img.convert("L"))
            g_img = self.resize(g_img.convert("RGB"))

            # Apply the exact same geometric augmentation to all modalities.
            if self.augment:
                if random.random() < 0.5:
                    v_img = TF.hflip(v_img)
                    i_img = TF.hflip(i_img)
                    g_img = TF.hflip(g_img)
                if random.random() < 0.2:
                    v_img = TF.vflip(v_img)
                    i_img = TF.vflip(i_img)
                    g_img = TF.vflip(g_img)

            v = self.to_tensor(v_img)
            ir = self.to_tensor(i_img)
            gt = self.to_tensor(g_img)
        return v, ir, gt


def match_pairs(vis_dir: str, ir_dir: str):
    vis = {p.name: p for p in get_all_images(vis_dir)}
    ir = {p.name: p for p in get_all_images(ir_dir)}
    names = sorted(set(vis) & set(ir))
    return [(str(vis[n]), str(ir[n]), n) for n in names]


class PairDataset(Dataset):
    def __init__(self, vis_dir: str, ir_dir: str, size: int = 256):
        self.samples = match_pairs(vis_dir, ir_dir)
        if len(self.samples) < 1:
            raise RuntimeError(
                "Need at least 1 matched VIS/IR pair with the same filename in test folders."
            )
        self.resize = T.Resize((size, size))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        vp, ip, name = self.samples[i]
        with Image.open(vp) as v_img, Image.open(ip) as i_img:
            v = self.to_tensor(self.resize(v_img.convert("RGB")))
            ir = self.to_tensor(self.resize(i_img.convert("L")))
        return v, ir, name


# =========================================================
# MODEL
# =========================================================
class CrossSRA(nn.Module):
    def __init__(self, dim: int, heads: int = 8, sr: int = 8):
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
            # kv_bias: [B, N_kv] -> [B, 1, 1, N_kv], shared across heads/query tokens.
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
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, vf, if_, h, w, vis_bias=None, ir_bias=None, attn_bias_gamma: float = 0.0):
        kv_bias_ir = None if ir_bias is None else (attn_bias_gamma * ir_bias)
        kv_bias_vis = None if vis_bias is None else (attn_bias_gamma * vis_bias)
        # Cross-attention: VIS tokens query IR tokens (vis -> ir).
        attn = torch.sigmoid(self.a) * self.v2i(vf, if_, h, w, kv_bias=kv_bias_ir)
        # Cross-attention: IR tokens query VIS tokens (ir -> vis).
        attn = attn + torch.sigmoid(self.b) * self.i2v(if_, vf, h, w, kv_bias=kv_bias_vis)
        vf = vf + attn
        vf = vf + self.ffn(self.norm(vf))
        return vf


def down(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, 2, 1), nn.BatchNorm2d(oc), nn.LeakyReLU(0.2, inplace=True)
    )


def up(ic, oc):
    return nn.Sequential(
        nn.ConvTranspose2d(ic, oc, 2, 2),
        nn.BatchNorm2d(oc),
        nn.LeakyReLU(0.2, inplace=True),
    )


def conv(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, 1, 1), nn.BatchNorm2d(oc), nn.LeakyReLU(0.2, inplace=True)
    )


class FusionNet(nn.Module):
    def __init__(
        self,
        base_ch: int = 32,
        bottleneck_ch: int = 256,
        attn_heads: int = 16,
        attn_sr: int = 4,
        attn_depth: int = 3,
        attn_mlp_ratio: float = 2.0,
        gate_alpha_vis: float = 0.25,
        gate_alpha_ir: float = 0.35,
        attn_bias_gamma: float = 0.20,
        attn_bias_detach: bool = True,
        dark_thresh: float = 0.45,
        dark_gain: float = 10.0,
        vis_conf_thresh: float = 0.55,
        vis_conf_gain: float = 11.0,
        ir_adv_margin: float = 0.00,
        sat_thresh: float = 0.98,
        sat_gain: float = 24.0,
        sat_ir_min: float = 0.50,
        quality_temp: float = 0.15,
        vis_quality_luma_w: float = 0.55,
        vis_quality_grad_w: float = 0.25,
        vis_quality_kp_w: float = 0.20,
        ir_quality_grad_w: float = 0.55,
        ir_quality_kp_w: float = 0.45,
        ir_priority_min: float = 0.01,
        ir_priority_max: float = 0.95,
        luma_pred_mix: float = 0.08,
        pred_rgb_mix: float = 0.05,
    ):
        super().__init__()
        if bottleneck_ch % attn_heads != 0:
            raise ValueError("bottleneck_ch must be divisible by attn_heads")

        self.v1, self.v2, self.v3 = down(3, base_ch), down(base_ch, base_ch * 2), down(base_ch * 2, bottleneck_ch)
        self.i1, self.i2, self.i3 = down(1, base_ch), down(base_ch, base_ch * 2), down(base_ch * 2, bottleneck_ch)

        self.fusion_blocks = nn.ModuleList(
            [
                FusionTokenBlock(
                    dim=bottleneck_ch,
                    heads=attn_heads,
                    sr=attn_sr,
                    mlp_ratio=attn_mlp_ratio,
                )
                for _ in range(attn_depth)
            ]
        )

        self.up3 = up(bottleneck_ch, base_ch * 2)
        self.c3 = conv((base_ch * 2) + (base_ch * 2) + (base_ch * 2), base_ch * 2)
        self.up2 = up(base_ch * 2, base_ch)
        self.c2 = conv(base_ch + base_ch + base_ch, base_ch)
        self.up1 = up(base_ch, base_ch)
        self.out = nn.Conv2d(base_ch, 3, 3, 1, 1)
        self.gate_alpha_vis = gate_alpha_vis
        self.gate_alpha_ir = gate_alpha_ir
        self.attn_bias_gamma = attn_bias_gamma
        self.attn_bias_detach = attn_bias_detach
        self.dark_thresh = dark_thresh
        self.dark_gain = dark_gain
        self.vis_conf_thresh = vis_conf_thresh
        self.vis_conf_gain = vis_conf_gain
        self.ir_adv_margin = ir_adv_margin
        self.sat_thresh = sat_thresh
        self.sat_gain = sat_gain
        self.sat_ir_min = sat_ir_min
        self.quality_temp = quality_temp
        self.vis_quality_luma_w = vis_quality_luma_w
        self.vis_quality_grad_w = vis_quality_grad_w
        self.vis_quality_kp_w = vis_quality_kp_w
        self.ir_quality_grad_w = ir_quality_grad_w
        self.ir_quality_kp_w = ir_quality_kp_w
        self.ir_priority_min = ir_priority_min
        self.ir_priority_max = ir_priority_max
        self.luma_pred_mix = luma_pred_mix
        self.pred_rgb_mix = pred_rgb_mix

    def _compute_ir_priority(self, v, i, vis_kmap=None, ir_kmap=None):
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

        vis_quality = (
            (self.vis_quality_luma_w * vis_y)
            + (self.vis_quality_grad_w * vis_grad)
            + (self.vis_quality_kp_w * vis_k)
        )
        ir_quality = (self.ir_quality_grad_w * ir_grad) + (self.ir_quality_kp_w * ir_k)

        # IR is preferred only when VIS is unreliable (dark/low-confidence)
        # AND IR quality is better than VIS quality.
        dark_conf = torch.sigmoid((self.dark_thresh - vis_y) * self.dark_gain)
        low_vis_conf = torch.sigmoid((self.vis_conf_thresh - vis_quality) * self.vis_conf_gain)
        need_ir = torch.maximum(dark_conf, low_vis_conf)
        quality_delta = (ir_quality - vis_quality - self.ir_adv_margin) / max(self.quality_temp, 1e-4)
        ir_adv = torch.sigmoid(quality_delta)
        ir_priority = need_ir * ir_adv

        # Special case: RGB saturation (near-white clipped pixels).
        # In these regions, force a minimum IR contribution because VIS details
        # are likely lost due to sensor saturation.
        sat_conf = saturation_confidence_map(
            v,
            sat_thresh=self.sat_thresh,
            sat_gain=self.sat_gain,
        )
        ir_priority = torch.maximum(ir_priority, self.sat_ir_min * sat_conf)
        return ir_priority.clamp(self.ir_priority_min, self.ir_priority_max)

    def _adaptive_blend(self, v, i, pred_rgb, ir_priority):
        vis_y = rgb_to_luma(v)
        pred_y = rgb_to_luma(pred_rgb)
        target_y = ((1.0 - ir_priority) * vis_y) + (ir_priority * i)
        fused_y = ((1.0 - self.luma_pred_mix) * target_y) + (self.luma_pred_mix * pred_y)

        # Keep VIS chroma and shift luminance/contrast via the adaptive Y-channel blend.
        gain = (fused_y / (vis_y + 1e-4)).clamp(0.2, 3.0)
        vis_color_preserved = (v * gain).clamp(0.0, 1.0)
        fused_rgb = ((1.0 - self.pred_rgb_mix) * vis_color_preserved) + (self.pred_rgb_mix * pred_rgb)
        return fused_rgb.clamp(0.0, 1.0)

    def forward(self, v, i, vis_kmap=None, ir_kmap=None, return_aux: bool = False):
        v1 = self.v1(v)
        v2 = self.v2(v1)
        v3 = self.v3(v2)

        i1 = self.i1(i)
        i2 = self.i2(i1)
        i3 = self.i3(i2)

        if vis_kmap is not None:
            vis_bias = F.interpolate(vis_kmap, size=v3.shape[-2:], mode="bilinear", align_corners=False)
            if self.attn_bias_detach:
                vis_bias = vis_bias.detach()
            # Feature gating: boost keypoint-relevant VIS regions.
            v3 = v3 * (1.0 + self.gate_alpha_vis * vis_bias)
        else:
            vis_bias = None

        if ir_kmap is not None:
            ir_bias = F.interpolate(ir_kmap, size=i3.shape[-2:], mode="bilinear", align_corners=False)
            if self.attn_bias_detach:
                ir_bias = ir_bias.detach()
            # Feature gating: boost keypoint-relevant IR regions.
            i3 = i3 * (1.0 + self.gate_alpha_ir * ir_bias)
        else:
            ir_bias = None

        bsz, c, h, w = v3.shape
        vf = v3.flatten(2).transpose(1, 2)
        if_ = i3.flatten(2).transpose(1, 2)

        f = vf
        for block in self.fusion_blocks:
            # Attention logit bias uses KeyNet maps as additive logits prior.
            f = block(
                f,
                if_,
                h,
                w,
                vis_bias=vis_bias,
                ir_bias=ir_bias,
                attn_bias_gamma=self.attn_bias_gamma,
            )
        f = f.transpose(1, 2).reshape(bsz, c, h, w)

        d3 = self.c3(torch.cat([self.up3(f), v2, i2], dim=1))
        d2 = self.c2(torch.cat([self.up2(d3), v1, i1], dim=1))
        d1 = self.up1(d2)
        pred_rgb = torch.sigmoid(self.out(d1))
        ir_priority = self._compute_ir_priority(v, i, vis_kmap=vis_kmap, ir_kmap=ir_kmap)
        fused = self._adaptive_blend(v, i, pred_rgb, ir_priority)
        if return_aux:
            return fused, {"ir_priority": ir_priority, "pred_rgb": pred_rgb}
        return fused


# =========================================================
# KEYNET RESPONSE LOSS
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
    def __init__(self, vis_weight: float = 1.0, ir_weight: float = 2.0):
        super().__init__()
        self.detector = KeyNetResponse()
        self.w_vis = vis_weight
        self.w_ir = ir_weight

    @staticmethod
    def _normalize_map(x: torch.Tensor) -> torch.Tensor:
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-6)

    def forward(self, v, i, f):
        detector_dtype = next(self.detector.parameters()).dtype
        v = v.mean(1, True).to(dtype=detector_dtype)
        i = i.to(dtype=detector_dtype)
        f = f.mean(1, True).to(dtype=detector_dtype)

        with torch.no_grad():
            rv = self.detector(v)
            ri = self.detector(i)
        rf = self.detector(f)

        if rf.shape[-2:] != rv.shape[-2:]:
            rf = F.interpolate(rf, size=rv.shape[-2:], mode="bilinear", align_corners=False)

        # Normalize each response map to reduce VIS amplitude dominance.
        rv = self._normalize_map(rv)
        ri = self._normalize_map(ri)
        rf = self._normalize_map(rf)

        vis_term = F.l1_loss(rf, rv)
        ir_term = F.l1_loss(rf, ri)
        return (self.w_vis * vis_term) + (self.w_ir * ir_term)


# =========================================================
# OTHER LOSSES
# =========================================================
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
    def __init__(
        self,
        w_gt_l1: float = 1.0,
        w_ssim: float = 0.9,
        w_grad: float = 0.7,
        w_src_l1: float = 0.25,
        w_sp: float = 0.08,
        sp_vis_weight: float = 1.0,
        sp_ir_weight: float = 2.0,
        gt_anchor: float = 0.08,
        w_vis_luma: float = 1.25,
        w_ir_luma: float = 0.80,
        w_vis_contrast: float = 0.85,
    ):
        super().__init__()
        self.sp = KeyNetLoss(vis_weight=sp_vis_weight, ir_weight=sp_ir_weight)
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

    def forward(self, v, i, gt, f, ir_priority=None):
        if ir_priority is None:
            ir_priority = torch.full_like(i, 0.5)
        vis_priority = 1.0 - ir_priority

        src_target = torch.max(v, i.repeat(1, 3, 1, 1))
        gt_l1 = F.l1_loss(f, gt)
        src_l1 = F.l1_loss(f, src_target)

        ssim_term = K.losses.ssim_loss(f, gt, window_size=11)
        if ssim_term.ndim > 0:
            ssim_term = ssim_term.mean()

        grad_term = F.l1_loss(
            self.sobel(f.mean(1, True)),
            self.sobel(gt.mean(1, True)),
        )
        gt_term = (
            (self.w_gt_l1 * gt_l1)
            + (self.w_ssim * ssim_term)
            + (self.w_grad * grad_term)
        )

        fused_y = rgb_to_luma(f)
        vis_y = rgb_to_luma(v)
        vis_luma_term = (vis_priority * (fused_y - vis_y).abs()).mean()
        ir_luma_term = (ir_priority * (fused_y - i).abs()).mean()

        fused_contrast = local_contrast_map(fused_y, kernel_size=5)
        vis_contrast = local_contrast_map(vis_y, kernel_size=5)
        vis_contrast_term = (vis_priority * (fused_contrast - vis_contrast).abs()).mean()

        sp_term = self.sp(v, i, f)
        return (
            self.gt_anchor * gt_term
            + self.w_src_l1 * src_l1
            + self.w_vis_luma * vis_luma_term
            + self.w_ir_luma * ir_luma_term
            + self.w_vis_contrast * vis_contrast_term
            + self.w_sp * sp_term
        )


# =========================================================
# TRAINING
# =========================================================
def _normalize_response_map(x: torch.Tensor) -> torch.Tensor:
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-6)


@torch.no_grad()
def compute_keypoint_maps(v: torch.Tensor, i: torch.Tensor, detector: nn.Module):
    """Compute normalized KeyNet response maps for VIS and IR inputs."""
    detector.eval()
    detector_dtype = next(detector.parameters()).dtype
    vg = v.mean(1, True).to(dtype=detector_dtype)
    ig = i.to(dtype=detector_dtype)
    rv = detector(vg)
    ri = detector(ig)
    rv = _normalize_response_map(rv)
    ri = _normalize_response_map(ri)
    return rv, ri


def _retention_from_responses(
    src_resp: torch.Tensor,
    fused_resp: torch.Tensor,
    topk: int,
    tolerance_px: int = 0,
) -> torch.Tensor:
    if fused_resp.shape[-2:] != src_resp.shape[-2:]:
        fused_resp = F.interpolate(
            fused_resp, size=src_resp.shape[-2:], mode="bilinear", align_corners=False
        )

    bsz, ch, h, w = src_resp.shape
    src_flat = src_resp.flatten(1)
    fused_flat = fused_resp.flatten(1)
    k = min(topk, src_flat.shape[1], fused_flat.shape[1])
    if k < 1:
        raise ValueError("topk must be >= 1 for keypoint retention.")

    src_idx = src_flat.topk(k=k, dim=1).indices
    fused_idx = fused_flat.topk(k=k, dim=1).indices

    src_mask = torch.zeros_like(src_flat, dtype=torch.float32)
    fused_mask = torch.zeros_like(fused_flat, dtype=torch.float32)
    src_mask.scatter_(1, src_idx, 1.0)
    fused_mask.scatter_(1, fused_idx, 1.0)

    src_mask = src_mask.view(bsz, ch, h, w)
    fused_mask = fused_mask.view(bsz, ch, h, w)

    # Neighborhood-tolerant overlap: accept fused keypoints within a
    # (2 * tolerance_px + 1) window around each source keypoint.
    if tolerance_px > 0:
        kernel = (2 * tolerance_px) + 1
        fused_mask = F.max_pool2d(
            fused_mask,
            kernel_size=kernel,
            stride=1,
            padding=tolerance_px,
        )

    inter = (src_mask * (fused_mask > 0).float()).flatten(1).sum(dim=1)
    denom = src_mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return inter / denom


@torch.no_grad()
def evaluate_keypoint_retention(
    net,
    detector,
    loader,
    device,
    use_amp,
    topk: int,
    max_batches: int,
    tolerance_px: int,
):
    autocast_dev = "cuda" if device.type == "cuda" else "cpu"
    net.eval()
    detector.eval()
    detector_dtype = next(detector.parameters()).dtype

    sum_vis = 0.0
    sum_ir = 0.0
    n = 0

    for bi, (v, i, _gt) in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break

        v = v.to(device, non_blocking=True)
        i = i.to(device, non_blocking=True)
        with torch.amp.autocast(autocast_dev, enabled=use_amp):
            vis_kmap, ir_kmap = compute_keypoint_maps(v, i, detector)
            f = net(v, i, vis_kmap=vis_kmap, ir_kmap=ir_kmap)

        # KeyNet expects fp32-style inputs; fused output may be fp16 from autocast.
        fg = f.mean(1, True).to(dtype=detector_dtype)
        rv = vis_kmap.to(dtype=detector_dtype)
        ri = ir_kmap.to(dtype=detector_dtype)
        rf = detector(fg)

        # Response normalization reduces VIS intensity dominance.
        rf = _normalize_response_map(rf)

        vis_ret = _retention_from_responses(rv, rf, topk=topk, tolerance_px=tolerance_px)
        ir_ret = _retention_from_responses(ri, rf, topk=topk, tolerance_px=tolerance_px)

        sum_vis += vis_ret.sum().item()
        sum_ir += ir_ret.sum().item()
        n += v.shape[0]

    if n == 0:
        return 0.0, 0.0, 0.0

    vis_score = sum_vis / n
    ir_score = sum_ir / n
    mean_score = 0.5 * (vis_score + ir_score)
    return vis_score, ir_score, mean_score


def _compute_split_counts(n_samples: int, train_ratio: float, val_ratio: float, test_ratio: float):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")
    if n_samples < 3:
        raise RuntimeError("Need at least 3 samples to create train/val/test splits.")

    n_train = max(1, int(n_samples * train_ratio))
    n_val = max(1, int(n_samples * val_ratio))
    n_test = n_samples - n_train - n_val

    # Keep at least one sample in each split for very small datasets.
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
    vis_dir,
    ir_dir,
    gt_dir,
    size,
    batch_size,
    workers,
    seed,
    train_ratio,
    val_ratio,
    test_ratio,
    train_aug=True,
):
    samples = match_triplets(vis_dir, ir_dir, gt_dir)
    n_train, n_val, n_test = _compute_split_counts(
        len(samples),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(samples), generator=gen).tolist()
    tr_ids = order[:n_train]
    vl_ids = order[n_train : n_train + n_val]
    te_ids = order[n_train + n_val :]
    tr_ds = FusionDataset(samples, size=size, augment=train_aug, indices=tr_ids)
    vl_ds = FusionDataset(samples, size=size, augment=False, indices=vl_ids)
    te_ds = FusionDataset(samples, size=size, augment=False, indices=te_ids)

    pin = torch.cuda.is_available()
    tr_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
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


@torch.no_grad()
def save_epoch_previews(
    net,
    detector,
    loader,
    device,
    use_amp,
    out_root,
    epoch,
    max_items,
    include_gt=True,
):
    if max_items <= 0:
        return

    epoch_dir = Path(out_root) / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    autocast_dev = "cuda" if device.type == "cuda" else "cpu"

    ds_len = len(loader.dataset) if hasattr(loader, "dataset") else None
    # Use a rolling contiguous preview window so each epoch visualizes
    # different test samples instead of repeatedly saving the first batch.
    if ds_len is not None and ds_len > 0:
        target_count = min(max_items, ds_len)
        if ds_len <= target_count:
            start_idx = 0
        else:
            window = ds_len - target_count + 1
            start_idx = ((epoch - 1) * target_count) % window
        end_idx = start_idx + target_count
    else:
        start_idx = 0
        end_idx = max_items

    net.eval()
    saved = 0
    sample_idx = 0
    for batch in loader:
        if include_gt:
            v, i, gt = batch
            names = None
        else:
            v, i, names = batch
            gt = None

        v = v.to(device, non_blocking=True)
        i = i.to(device, non_blocking=True)
        vis_kmap, ir_kmap = compute_keypoint_maps(v, i, detector)
        with torch.amp.autocast(autocast_dev, enabled=use_amp):
            f = net(v, i, vis_kmap=vis_kmap, ir_kmap=ir_kmap)

        v = v.detach().cpu()
        i = i.detach().cpu()
        f = f.detach().cpu().clamp_(0.0, 1.0)
        if gt is not None:
            gt = gt.detach().cpu()

        bsz = f.shape[0]
        for bi in range(bsz):
            in_preview_window = (start_idx <= sample_idx) and (sample_idx < end_idx)
            sample_idx += 1
            if not in_preview_window:
                continue
            if saved >= max_items:
                break

            if names is None:
                stem = f"sample_{saved + 1:04d}"
            else:
                stem = Path(str(names[bi])).stem or f"sample_{saved + 1:04d}"

            fused_path = epoch_dir / f"{stem}_fused.png"
            panel_path = epoch_dir / f"{stem}_panel.png"

            ir_rgb = i[bi].repeat(3, 1, 1)
            panes = [v[bi], ir_rgb, f[bi]]
            if gt is not None:
                panes.append(gt[bi])
            panel = torch.cat(panes, dim=2)

            vutils.save_image(f[bi], str(fused_path))
            vutils.save_image(panel, str(panel_path))
            saved += 1

        if saved >= max_items:
            break

    print(
        f"Saved {saved} preview sample(s) to: {epoch_dir} "
        f"(test indices {start_idx}..{max(start_idx, end_idx - 1)})"
    )


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
    )
    print(
        f"Matched samples: {n_samples} | "
        f"train: {n_train} ({len(tr_loader)} batches) | "
        f"val: {n_val} ({len(vl_loader)} batches) | "
        f"test: {n_test} ({len(te_loader)} batches)"
    )
    print("Preview source: internal 5% test split from the same dataset")
    retention_loader = vl_loader if args.retention_split == "val" else te_loader
    print(f"Retention source: internal {args.retention_split} split")

    net = FusionNet(
        base_ch=args.base_ch,
        bottleneck_ch=args.bottleneck_ch,
        attn_heads=args.attn_heads,
        attn_sr=args.attn_sr,
        attn_depth=args.attn_depth,
        attn_mlp_ratio=args.attn_mlp_ratio,
        gate_alpha_vis=args.gate_alpha_vis,
        gate_alpha_ir=args.gate_alpha_ir,
        attn_bias_gamma=args.attn_bias_gamma,
        attn_bias_detach=args.attn_bias_detach,
        dark_thresh=args.dark_thresh,
        dark_gain=args.dark_gain,
        vis_conf_thresh=args.vis_conf_thresh,
        vis_conf_gain=args.vis_conf_gain,
        ir_adv_margin=args.ir_adv_margin,
        quality_temp=args.quality_temp,
        vis_quality_luma_w=args.vis_quality_luma_w,
        vis_quality_grad_w=args.vis_quality_grad_w,
        vis_quality_kp_w=args.vis_quality_kp_w,
        ir_quality_grad_w=args.ir_quality_grad_w,
        ir_quality_kp_w=args.ir_quality_kp_w,
        ir_priority_min=args.ir_priority_min,
        ir_priority_max=args.ir_priority_max,
        luma_pred_mix=args.luma_pred_mix,
        pred_rgb_mix=args.pred_rgb_mix,
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
    ).to(device)
    loss_fn.sp.eval()
    opt = torch.optim.AdamW(
        net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=args.lr_decay,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = float("inf")
    cnt = 0
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    os.makedirs(args.preview_dir, exist_ok=True)
    base_w_sp = args.w_sp
    base_sp_ir_weight = args.sp_ir_weight
    base_sp_vis_weight = args.sp_vis_weight
    prev_ir_ret = None
    prev_vis_ret = None
    best_vloss = float("inf")
    ema_ir_gain = 0.0
    ema_vis_gain = 0.0
    bad_vloss_streak = 0


    for e in range(args.epochs):
        epoch_start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch_start = time.perf_counter()
        print(f"Epoch {e + 1}/{args.epochs} | start: {epoch_start_ts}")
        net.train()
        tl = 0.0
        train_ir_priority = 0.0
        train_ir_takeover = 0.0
        for v, i, gt in tr_loader:
            v = v.to(device, non_blocking=True)
            i = i.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(autocast_dev, enabled=use_amp):
                vis_kmap, ir_kmap = compute_keypoint_maps(v, i, loss_fn.sp.detector)
                f, aux = net(v, i, vis_kmap=vis_kmap, ir_kmap=ir_kmap, return_aux=True)
                loss = loss_fn(v, i, gt, f, ir_priority=aux["ir_priority"])

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
            train_ir_priority += aux["ir_priority"].mean().item()
            train_ir_takeover += (aux["ir_priority"] > 0.5).float().mean().item()

        net.eval()
        vloss = 0.0
        val_ir_priority = 0.0
        val_ir_takeover = 0.0
        with torch.no_grad():
            for v, i, gt in vl_loader:
                v = v.to(device, non_blocking=True)
                i = i.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                with torch.amp.autocast(autocast_dev, enabled=use_amp):
                    vis_kmap, ir_kmap = compute_keypoint_maps(v, i, loss_fn.sp.detector)
                    pred, aux = net(v, i, vis_kmap=vis_kmap, ir_kmap=ir_kmap, return_aux=True)
                    vloss += loss_fn(v, i, gt, pred, ir_priority=aux["ir_priority"]).item()
                val_ir_priority += aux["ir_priority"].mean().item()
                val_ir_takeover += (aux["ir_priority"] > 0.5).float().mean().item()

        tl /= max(1, len(tr_loader))
        vloss /= max(1, len(vl_loader))
        train_ir_priority /= max(1, len(tr_loader))
        train_ir_takeover /= max(1, len(tr_loader))
        val_ir_priority /= max(1, len(vl_loader))
        val_ir_takeover /= max(1, len(vl_loader))
        scheduler.step(vloss)
        print(f"TL={tl:.4f} VL={vloss:.4f} LR={opt.param_groups[0]['lr']:.2e}")
        print(
            "Modality gate | "
            f"train_ir_priority={train_ir_priority:.3f} train_ir_takeover={train_ir_takeover:.3f} "
            f"val_ir_priority={val_ir_priority:.3f} val_ir_takeover={val_ir_takeover:.3f}"
        )

        vis_ret, ir_ret, mean_ret = evaluate_keypoint_retention(
            net=net,
            detector=loss_fn.sp.detector,
            loader=retention_loader,
            device=device,
            use_amp=use_amp,
            topk=args.retention_topk,
            max_batches=args.retention_eval_batches,
            tolerance_px=args.retention_tolerance_px,
        )
        print(
            "Keypoint retention "
            f"(vis->fused={vis_ret:.4f}, ir->fused={ir_ret:.4f}, mean={mean_ret:.4f})"
        )

        if vloss < best:
            best = vloss
            cnt = 0
            torch.save(net.state_dict(), args.ckpt)
            print(f"Saved best model to: {args.ckpt}")
        else:
            cnt += 1

        if vloss + args.vloss_guard_tol < best_vloss:
            best_vloss = vloss
            bad_vloss_streak = 0
        elif vloss > best_vloss + args.vloss_guard_tol:
            bad_vloss_streak += 1

        save_epoch_previews(
            net=net,
            detector=loss_fn.sp.detector,
            loader=te_loader,
            device=device,
            use_amp=use_amp,
            out_root=args.preview_dir,
            epoch=e + 1,
            max_items=args.preview_count,
            include_gt=True,
        )

        # Adaptive weighting for next epoch with dual gates:
        # - vis_push: recover VIS retention / protect floor
        # - ir_push: lift IR retention when VIS remains healthy
        err_ir = args.retention_target_ir - ir_ret
        err_vis = args.retention_target_vis - vis_ret

        ir_gain = 0.0 if prev_ir_ret is None else (ir_ret - prev_ir_ret)
        vis_gain = 0.0 if prev_vis_ret is None else (vis_ret - prev_vis_ret)
        prev_ir_ret = ir_ret
        prev_vis_ret = vis_ret

        ema_ir_gain = (args.retention_ema_beta * ema_ir_gain) + ((1.0 - args.retention_ema_beta) * ir_gain)
        ema_vis_gain = (args.retention_ema_beta * ema_vis_gain) + ((1.0 - args.retention_ema_beta) * vis_gain)

        loss_guard_block = bad_vloss_streak >= args.vloss_guard_patience
        ir_trend_healthy = ema_ir_gain >= args.retention_ir_min_gain
        vis_push = (
            (vis_ret < (args.retention_target_vis - args.retention_margin_vis))
            or (vis_ret < args.retention_vis_floor)
            or (ema_vis_gain < max(args.vis_gain_floor, args.retention_vis_min_gain))
        )
        ir_push = (
            (ir_ret < (args.retention_target_ir - args.retention_margin_ir))
            and (vis_ret >= args.retention_vis_floor)
            and (ema_ir_gain >= args.ir_gain_floor)
            and not loss_guard_block
        )

        src_scale = 1.0
        if vis_push:
            vis_boost = min(1.0, max(0.0, args.retention_target_vis - vis_ret))
            vis_step = args.retention_weight_lr_vis * args.vis_push_step * vis_boost
            loss_fn.sp.w_vis = float(min(2.0, max(args.sp_vis_weight_min, loss_fn.sp.w_vis + vis_step)))
            src_scale += args.src_l1_vis_boost * vis_boost
        else:
            over_base_vis = max(0.0, loss_fn.sp.w_vis - base_sp_vis_weight)
            loss_fn.sp.w_vis = float(
                max(args.sp_vis_weight_min, loss_fn.sp.w_vis - args.vis_push_decay * over_base_vis)
            )

        if ir_push:
            ir_boost = min(1.0, max(0.0, args.retention_target_ir - ir_ret))
            ir_step = args.retention_weight_lr_ir * args.ir_push_step * ir_boost
            loss_fn.sp.w_ir = float(min(args.sp_ir_weight_max, max(loss_fn.sp.w_vis, loss_fn.sp.w_ir + ir_step)))
            sp_delta = args.w_sp_step * ir_boost if ir_trend_healthy else 0.0
            src_scale += args.src_l1_ir_boost * ir_boost
        else:
            over_base_ir = max(0.0, loss_fn.sp.w_ir - base_sp_ir_weight)
            loss_fn.sp.w_ir = float(max(loss_fn.sp.w_vis, loss_fn.sp.w_ir - args.sp_ir_decay * over_base_ir))
            over_base_sp = max(0.0, loss_fn.w_sp - base_w_sp)
            sp_delta = -args.w_sp_decay * over_base_sp

        if loss_guard_block:
            sp_delta *= args.loss_guard_sp_damp
            src_scale = min(src_scale, 1.0)

        loss_fn.w_sp = float(min(args.w_sp_max, max(args.w_sp_min, loss_fn.w_sp + sp_delta)))
        loss_fn.w_src_l1 = float(
            min(args.w_src_l1_max, max(args.w_src_l1_min, loss_fn.w_src_l1 * src_scale))
        )

        print(
            "Adaptive control | "
            f"vis_push={vis_push} ir_push={ir_push} "
            f"ir_trend_healthy={ir_trend_healthy} "
            f"loss_guard_block={loss_guard_block} bad_vloss_streak={bad_vloss_streak}"
        )
        print(
            "Adaptive metrics | "
            f"err_vis={err_vis:.4f} err_ir={err_ir:.4f} "
            f"vis_gain={vis_gain:.4f} ir_gain={ir_gain:.4f} "
            f"ema_vis_gain={ema_vis_gain:.4f} ema_ir_gain={ema_ir_gain:.4f}"
        )
        print(
            "Adaptive weights | "
            f"w_sp={loss_fn.w_sp:.4f} w_src_l1={loss_fn.w_src_l1:.4f} "
            f"sp_vis_w={loss_fn.sp.w_vis:.3f} sp_ir_w={loss_fn.sp.w_ir:.3f} "
            f"sp_delta={sp_delta:.5f} src_scale={src_scale:.4f}"
        )
        epoch_secs = time.perf_counter() - epoch_start
        total_secs = time.perf_counter() - run_start
        print(
            f"Epoch {e + 1} time: {epoch_secs:.1f}s ({epoch_secs / 60.0:.2f} min) | "
            f"total elapsed: {total_secs / 60.0:.2f} min"
        )

        if cnt >= args.patience:
            print("Early stopping")
            break


def parse_args():
    p = argparse.ArgumentParser("KPFuse trainer")
    p.add_argument("--vis-dir", type=str, required=True, help="Visible image folder")
    p.add_argument("--ir-dir", type=str, required=True, help="Infrared image folder")
    p.add_argument("--gt-dir", type=str, required=True, help="Ground-truth fused image folder")
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
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", type=str, default="best_kpfuse.pth")
    p.add_argument("--preview-dir", type=str, default="kpfuse_previews")
    p.add_argument("--preview-count", type=int, default=6)
    p.add_argument("--retention-split", type=str, choices=["val", "test"], default="test")
    p.add_argument("--retention-topk", type=int, default=180)
    p.add_argument(
        "--retention-tolerance-px",
        type=int,
        default=3,
        help="Neighborhood tolerance (pixels) for retention overlap via dilation.",
    )
    p.add_argument(
        "--retention-eval-batches",
        type=int,
        default=16,
        help="Max batches for per-epoch retention eval (-1 for full split).",
    )
    p.add_argument("--retention-target-vis", type=float, default=0.62)
    p.add_argument("--retention-target-ir", type=float, default=0.20)
    p.add_argument(
        "--retention-vis-floor",
        type=float,
        default=0.58,
        help="Protect VIS retention by damping IR-push updates below this floor.",
    )
    p.add_argument(
        "--retention-weight-lr-ir",
        type=float,
        default=0.65,
        help="Aggressiveness for IR-retention driven adaptation.",
    )
    p.add_argument(
        "--retention-weight-lr-vis",
        type=float,
        default=0.25,
        help="Aggressiveness for VIS-retention balancing/protection.",
    )
    p.add_argument(
        "--retention-ir-min-gain",
        type=float,
        default=0.0,
        help="Minimum IR-retention gain required to keep increasing w_sp.",
    )
    p.add_argument(
        "--retention-vis-min-gain",
        type=float,
        default=-0.001,
        help="Minimum VIS-retention gain considered as healthy trend.",
    )
    p.add_argument(
        "--retention-ema-beta",
        type=float,
        default=0.9,
        help="EMA smoothing factor for retention gains (0-1, higher = smoother).",
    )
    p.add_argument(
        "--retention-margin-vis",
        type=float,
        default=0.01,
        help="Activation margin below VIS target for vis_push.",
    )
    p.add_argument(
        "--retention-margin-ir",
        type=float,
        default=0.01,
        help="Activation margin below IR target for ir_push.",
    )
    p.add_argument(
        "--vis-gain-floor",
        type=float,
        default=-0.0010,
        help="Minimum EMA VIS gain before triggering vis_push trend recovery.",
    )
    p.add_argument(
        "--ir-gain-floor",
        type=float,
        default=-0.0005,
        help="Minimum EMA IR gain required to keep IR push active.",
    )
    p.add_argument(
        "--vloss-guard-patience",
        type=int,
        default=3,
        help="Consecutive epochs of rising VL before adaptation is damped.",
    )
    p.add_argument(
        "--vloss-guard-tol",
        type=float,
        default=0.0015,
        help="Minimum VL increase to count toward loss guard.",
    )
    p.add_argument(
        "--sp-vis-weight-min",
        type=float,
        default=0.8,
        help="Lower bound for VIS keypoint branch weight during adaptation.",
    )
    p.add_argument(
        "--vis-push-step",
        type=float,
        default=0.02,
        help="Step size for increasing VIS keypoint branch when vis_push is active.",
    )
    p.add_argument(
        "--ir-push-step",
        type=float,
        default=0.03,
        help="Step size for increasing IR keypoint branch when ir_push is active.",
    )
    p.add_argument(
        "--vis-push-decay",
        type=float,
        default=0.08,
        help="Decay rate for VIS keypoint branch when vis_push is inactive.",
    )
    p.add_argument(
        "--src-l1-vis-boost",
        type=float,
        default=0.0,
        help="Multiplicative source-L1 boost factor when vis_push is active.",
    )
    p.add_argument(
        "--src-l1-ir-boost",
        type=float,
        default=0.05,
        help="Multiplicative source-L1 boost factor when ir_push is active.",
    )
    p.add_argument(
        "--loss-guard-sp-damp",
        type=float,
        default=0.5,
        help="Damping factor applied to SP-weight updates during validation-loss guard.",
    )
    p.add_argument("--train-aug", dest="train_aug", action="store_true")
    p.add_argument("--no-train-aug", dest="train_aug", action="store_false")
    p.set_defaults(train_aug=True)
    # Rebalanced defaults after switching SSIM/gradient supervision to GT.
    p.add_argument("--w-gt-l1", type=float, default=1.0)
    p.add_argument("--w-ssim", type=float, default=0.9)
    p.add_argument("--w-grad", type=float, default=0.7)
    p.add_argument(
        "--gt-anchor",
        type=float,
        default=0.08,
        help="Scales total GT supervision so GT acts as a loose reference.",
    )
    p.add_argument("--w-src-l1", type=float, default=0.25)
    p.add_argument(
        "--w-vis-luma",
        type=float,
        default=1.25,
        help="Penalty to keep fused luminance close to VIS where VIS is trusted.",
    )
    p.add_argument(
        "--w-ir-luma",
        type=float,
        default=0.8,
        help="Penalty to keep fused luminance close to IR where IR is prioritized.",
    )
    p.add_argument(
        "--w-vis-contrast",
        type=float,
        default=0.85,
        help="Penalty to preserve VIS local contrast in VIS-priority regions.",
    )
    p.add_argument("--w-src-l1-min", type=float, default=0.05)
    p.add_argument("--w-src-l1-max", type=float, default=0.45)
    p.add_argument("--w-sp", type=float, default=0.08)
    p.add_argument("--w-sp-min", type=float, default=0.02)
    p.add_argument("--w-sp-max", type=float, default=0.2)
    p.add_argument(
        "--w-sp-step",
        type=float,
        default=0.03,
        help="Additive step size for w_sp when IR-retention improvement is allowed.",
    )
    p.add_argument(
        "--w-sp-decay",
        type=float,
        default=0.12,
        help="Decay rate for w_sp toward baseline when IR push is not allowed.",
    )
    p.add_argument("--sp-vis-weight", type=float, default=1.0)
    p.add_argument("--sp-ir-weight", type=float, default=2.0)
    p.add_argument("--sp-ir-weight-max", type=float, default=3.0)
    p.add_argument(
        "--sp-ir-decay",
        type=float,
        default=0.2,
        help="Decay rate for IR keypoint branch weight when IR push is disabled.",
    )
    # Capacity scaling knobs for higher GPU utilization.
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--bottleneck-ch", type=int, default=256)
    p.add_argument("--attn-heads", type=int, default=16)
    p.add_argument("--attn-sr", type=int, default=4)
    p.add_argument("--attn-depth", type=int, default=3)
    p.add_argument("--attn-mlp-ratio", type=float, default=2.0)
    p.add_argument(
        "--gate-alpha-vis",
        type=float,
        default=0.25,
        help="Strength of VIS feature gating from KeyNet heatmap.",
    )
    p.add_argument(
        "--gate-alpha-ir",
        type=float,
        default=0.35,
        help="Strength of IR feature gating from KeyNet heatmap.",
    )
    p.add_argument(
        "--attn-bias-gamma",
        type=float,
        default=0.20,
        help="Strength of KeyNet-derived additive bias on attention logits.",
    )
    p.add_argument(
        "--dark-thresh",
        type=float,
        default=0.45,
        help="VIS luminance threshold below which IR gets stronger priority.",
    )
    p.add_argument(
        "--dark-gain",
        type=float,
        default=10.0,
        help="Steepness of low-light IR-priority activation.",
    )
    p.add_argument(
        "--vis-conf-thresh",
        type=float,
        default=0.55,
        help="VIS quality threshold below which IR fallback can activate.",
    )
    p.add_argument(
        "--vis-conf-gain",
        type=float,
        default=11.0,
        help="Steepness of VIS low-confidence activation for IR fallback.",
    )
    p.add_argument(
        "--ir-adv-margin",
        type=float,
        default=0.00,
        help="Required IR quality advantage before IR takes precedence.",
    )
    p.add_argument(
        "--sat-thresh",
        type=float,
        default=0.98,
        help="RGB saturation threshold in [0,1] (e.g., 250/255 ~= 0.98).",
    )
    p.add_argument(
        "--sat-gain",
        type=float,
        default=24.0,
        help="Steepness of saturation-confidence activation.",
    )
    p.add_argument(
        "--sat-ir-min",
        type=float,
        default=0.50,
        help="Minimum IR priority enforced in saturated RGB regions.",
    )
    p.add_argument(
        "--quality-temp",
        type=float,
        default=0.15,
        help="Temperature for IR-vs-VIS quality comparison in priority gate.",
    )
    p.add_argument("--vis-quality-luma-w", type=float, default=0.55)
    p.add_argument("--vis-quality-grad-w", type=float, default=0.25)
    p.add_argument("--vis-quality-kp-w", type=float, default=0.20)
    p.add_argument("--ir-quality-grad-w", type=float, default=0.55)
    p.add_argument("--ir-quality-kp-w", type=float, default=0.45)
    p.add_argument("--ir-priority-min", type=float, default=0.01)
    p.add_argument("--ir-priority-max", type=float, default=0.95)
    p.add_argument(
        "--luma-pred-mix",
        type=float,
        default=0.06,
        help="Blend ratio between adaptive source luminance and decoder luminance.",
    )
    p.add_argument(
        "--pred-rgb-mix",
        type=float,
        default=0.04,
        help="Blend ratio between VIS-color-preserved output and decoder RGB output.",
    )
    p.add_argument(
        "--attn-bias-detach",
        dest="attn_bias_detach",
        action="store_true",
        help="Detach attention bias maps from gradients (recommended).",
    )
    p.add_argument(
        "--no-attn-bias-detach",
        dest="attn_bias_detach",
        action="store_false",
    )
    p.set_defaults(attn_bias_detach=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-8:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio must equal 1.0")
    if args.retention_topk < 1:
        raise ValueError("--retention-topk must be >= 1")
    if args.retention_tolerance_px < 0:
        raise ValueError("--retention-tolerance-px must be >= 0")
    if args.retention_eval_batches == 0 or args.retention_eval_batches < -1:
        raise ValueError("--retention-eval-batches must be -1 or a positive integer")
    if args.retention_ir_min_gain < 0:
        raise ValueError("--retention-ir-min-gain must be >= 0")
    if not (0.0 <= args.retention_ema_beta < 1.0):
        raise ValueError("--retention-ema-beta must satisfy 0 <= beta < 1")
    if args.vloss_guard_patience < 1:
        raise ValueError("--vloss-guard-patience must be >= 1")
    if args.vloss_guard_tol < 0:
        raise ValueError("--vloss-guard-tol must be >= 0")
    if args.sp_vis_weight_min <= 0:
        raise ValueError("--sp-vis-weight-min must be > 0")
    if not (0.0 <= args.gt_anchor <= 1.0):
        raise ValueError("--gt-anchor must be in [0, 1]")
    if args.quality_temp <= 0:
        raise ValueError("--quality-temp must be > 0")
    if args.dark_gain <= 0:
        raise ValueError("--dark-gain must be > 0")
    if args.sat_gain <= 0:
        raise ValueError("--sat-gain must be > 0")
    if not (0.0 <= args.sat_thresh <= 1.0):
        raise ValueError("--sat-thresh must be in [0, 1]")
    if not (0.0 <= args.sat_ir_min <= 1.0):
        raise ValueError("--sat-ir-min must be in [0, 1]")
    if not (0.0 <= args.ir_priority_min < args.ir_priority_max <= 1.0):
        raise ValueError("--ir-priority-min/max must satisfy 0<=min<max<=1")
    if not (0.0 <= args.luma_pred_mix <= 1.0):
        raise ValueError("--luma-pred-mix must be in [0, 1]")
    if not (0.0 <= args.pred_rgb_mix <= 1.0):
        raise ValueError("--pred-rgb-mix must be in [0, 1]")
    for req_path in [args.vis_dir, args.ir_dir, args.gt_dir]:
        if not os.path.isdir(req_path):
            raise FileNotFoundError(f"Missing directory: {req_path}")
    train(args)

# nohup python3 -u bidirectional_crossattn.py --vis-dir '/home/chandra/Documents/Mizzou/VIS_IR_Fusion/training_data/vi' --ir-dir '/home/chandra/Documents/Mizzou/VIS_IR_Fusion/training_data/ir' --gt-dir '/home/chandra/Documents/Mizzou/VIS_IR_Fusion/training_data/SGT_Direct_Fusion' --epochs 100 --batch-size 12 --size 512 --base-ch 48 --bottleneck-ch 192 --attn-depth 3 --attn-heads 8 --attn-sr 2  --workers 4 --ckpt kpfuse_v1.pth  > train_log_kpfuse.txt 2>&1 &
