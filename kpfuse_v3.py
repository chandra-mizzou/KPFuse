import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import kpfuse_v2 as v2


class FusionNet(nn.Module):
    """
    KPFuse v3 model:
    - 3-layer encoder/decoder (reduced from v2's 4 layers)
    - 3 outputs (VIS-dominant, IR-dominant, exclusive-weighted)
    - compatible with v2 train/eval pipeline
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
        gate_alpha_vis: float = 0.25,
        gate_alpha_ir: float = 0.35,
        attn_bias_gamma: float = 0.20,
        attn_bias_detach: bool = True,
        luma_pred_mix: float = 0.06,
        pred_rgb_mix: float = 0.04,
    ):
        super().__init__()
        ch1 = base_ch
        ch2 = base_ch * 2
        ch3 = bottleneck_ch

        # Reuse v2 argument names:
        # - attn_depth_l3/sr_l3 -> stage-2 fusion (mid-level)
        # - attn_depth_l4/sr_l4 -> stage-3 bottleneck fusion
        if (attn_depth_l3 > 0) and (ch2 % attn_heads != 0):
            raise ValueError("Layer-2 channels must be divisible by attn_heads when attn_depth_l3>0")
        if (attn_depth_l4 > 0) and (ch3 % attn_heads != 0):
            raise ValueError("Layer-3 channels must be divisible by attn_heads when attn_depth_l4>0")

        # 3-layer encoder
        self.v1 = v2.down(3, ch1)
        self.v2 = v2.down(ch1, ch2)
        self.v3 = v2.down(ch2, ch3)
        self.i1 = v2.down(1, ch1)
        self.i2 = v2.down(ch1, ch2)
        self.i3 = v2.down(ch2, ch3)

        self.fusion_blocks_l2 = nn.ModuleList(
            [
                v2.FusionTokenBlock(
                    dim=ch2,
                    heads=attn_heads,
                    sr=attn_sr_l3,
                    mlp_ratio=attn_mlp_ratio,
                )
                for _ in range(attn_depth_l3)
            ]
        )
        self.fusion_blocks_l3 = nn.ModuleList(
            [
                v2.FusionTokenBlock(
                    dim=ch3,
                    heads=attn_heads,
                    sr=attn_sr_l4,
                    mlp_ratio=attn_mlp_ratio,
                )
                for _ in range(attn_depth_l4)
            ]
        )

        # 3-layer decoder
        self.up3 = v2.up(ch3, ch2)
        self.c3 = v2.conv(ch2 + ch2 + ch2, ch2)
        self.up2 = v2.up(ch2, ch1)
        self.c2 = v2.conv(ch1 + ch1 + ch1, ch1)
        self.up1 = v2.up(ch1, ch1)
        self.out = nn.Conv2d(ch1, 3, 3, 1, 1)

        self.gate_alpha_vis = gate_alpha_vis
        self.gate_alpha_ir = gate_alpha_ir
        self.attn_bias_gamma = attn_bias_gamma
        self.attn_bias_detach = attn_bias_detach
        self.luma_pred_mix = luma_pred_mix
        self.pred_rgb_mix = pred_rgb_mix

    def _run_fusion_stage(
        self,
        vis_feat: torch.Tensor,
        ir_feat: torch.Tensor,
        blocks: nn.ModuleList,
        vis_kmap,
        ir_kmap,
    ) -> torch.Tensor:
        if len(blocks) == 0:
            return vis_feat

        vis_bias = None
        if vis_kmap is not None:
            vis_bias = F.interpolate(vis_kmap, size=vis_feat.shape[-2:], mode="bilinear", align_corners=False)
            if self.attn_bias_detach:
                vis_bias = vis_bias.detach()
            vis_feat = vis_feat * (1.0 + self.gate_alpha_vis * vis_bias)

        ir_bias = None
        if ir_kmap is not None:
            ir_bias = F.interpolate(ir_kmap, size=ir_feat.shape[-2:], mode="bilinear", align_corners=False)
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

    def _compute_exclusive_ir_weight(self, v, i, vis_kmap, ir_kmap):
        vis_y = v2.rgb_to_luma(v)
        vis_grad = v2.normalize_map_per_sample(v2.gradient_mag_map(vis_y))
        ir_grad = v2.normalize_map_per_sample(v2.gradient_mag_map(i))

        if vis_kmap is None:
            vis_k = vis_grad
        else:
            vis_k = F.interpolate(vis_kmap, size=vis_y.shape[-2:], mode="bilinear", align_corners=False)
            vis_k = v2.normalize_map_per_sample(vis_k)

        if ir_kmap is None:
            ir_k = ir_grad
        else:
            ir_k = F.interpolate(ir_kmap, size=i.shape[-2:], mode="bilinear", align_corners=False)
            ir_k = v2.normalize_map_per_sample(ir_k)

        vis_exclusive = F.relu(vis_grad - ir_grad) + F.relu(vis_k - ir_k)
        ir_exclusive = F.relu(ir_grad - vis_grad) + F.relu(ir_k - vis_k)
        ir_w = ir_exclusive / (vis_exclusive + ir_exclusive + 1e-6)
        return ir_w.clamp(0.0, 1.0)

    def _adaptive_blend(self, v, i, pred_rgb, ir_priority):
        vis_y = v2.rgb_to_luma(v)
        pred_y = v2.rgb_to_luma(pred_rgb)
        target_y = ((1.0 - ir_priority) * vis_y) + (ir_priority * i)
        fused_y = ((1.0 - self.luma_pred_mix) * target_y) + (self.luma_pred_mix * pred_y)
        gain = (fused_y / (vis_y + 1e-4)).clamp(0.2, 3.0)
        vis_color_preserved = (v * gain).clamp(0.0, 1.0)
        fused_rgb = ((1.0 - self.pred_rgb_mix) * vis_color_preserved) + (self.pred_rgb_mix * pred_rgb)
        return fused_rgb.clamp(0.0, 1.0)

    def forward(self, v, i, vis_kmap=None, ir_kmap=None, return_aux: bool = False, return_all: bool = False):
        v1 = self.v1(v)
        v2f = self.v2(v1)
        i1 = self.i1(i)
        i2f = self.i2(i1)

        v2f = self._run_fusion_stage(v2f, i2f, self.fusion_blocks_l2, vis_kmap, ir_kmap)

        v3 = self.v3(v2f)
        i3 = self.i3(i2f)
        v3 = self._run_fusion_stage(v3, i3, self.fusion_blocks_l3, vis_kmap, ir_kmap)

        u3 = self.up3(v3)
        if u3.shape[-2:] != v2f.shape[-2:]:
            u3 = F.interpolate(u3, size=v2f.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.c3(torch.cat([u3, v2f, i2f], dim=1))

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


def train(args):
    original = v2.FusionNet
    try:
        # Reuse the full v2 training pipeline with this v3 architecture.
        v2.FusionNet = FusionNet
        return v2.train(args)
    finally:
        v2.FusionNet = original


def parse_args():
    return v2.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-8:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio must equal 1.0")
    if args.train_views < 1:
        raise ValueError("--train-views must be >= 1")
    if args.attn_depth_l3 < 0 or args.attn_depth_l4 < 0:
        raise ValueError("--attn-depth-l3/l4 must be >= 0")
    if args.attn_sr_l3 < 1 or args.attn_sr_l4 < 1:
        raise ValueError("--attn-sr-l3/l4 must be >= 1")
    if min(args.loss_w_vis, args.loss_w_ir, args.loss_w_excl) < 0:
        raise ValueError("--loss-w-vis/--loss-w-ir/--loss-w-excl must be >= 0")
    if (args.loss_w_vis + args.loss_w_ir + args.loss_w_excl) <= 0:
        raise ValueError("At least one of --loss-w-vis/--loss-w-ir/--loss-w-excl must be > 0")
    if args.pad_multiple < 0:
        raise ValueError("--pad-multiple must be >= 0")
    if args.max_train_side < 0:
        raise ValueError("--max-train-side must be >= 0")
    if args.native_res_train and args.pad_multiple == 0:
        raise ValueError("--pad-multiple must be >= 1 when --native-res-train is enabled")
    for req in [args.vis_dir, args.ir_dir, args.gt_dir]:
        if not os.path.isdir(req):
            raise FileNotFoundError(f"Missing directory: {req}")
    train(args)
