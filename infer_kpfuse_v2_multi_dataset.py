import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from kpfuse_v2 import FusionNet, KeyNetResponse, compute_keypoint_maps


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def build_stem_map(files: Sequence[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in files:
        if p.stem not in out:
            out[p.stem] = p
    return out


def discover_datasets(datasets_root: Path, dataset_names: Sequence[str]) -> List[Path]:
    if dataset_names:
        cands = [datasets_root / d for d in dataset_names]
    else:
        cands = [p for p in sorted(datasets_root.iterdir()) if p.is_dir()]
    return [p for p in cands if (p / "vi").is_dir() and (p / "ir").is_dir()]


def load_pair(vis_path: Path, ir_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    with Image.open(vis_path) as v_img, Image.open(ir_path) as i_img:
        v = TF.to_tensor(v_img.convert("RGB"))
        i = TF.to_tensor(i_img.convert("L"))
    return v, i


def pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def unpad(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x


def parse_args():
    p = argparse.ArgumentParser("KPFuse v2 multi-dataset inference (3 outputs)")
    p.add_argument(
        "--datasets-root",
        type=str,
        required=True,
        help="Path to root containing dataset folders (e.g., dataset_21/datasets)",
    )
    p.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root. Saves <output-root>/<dataset>/{VIS-dominant,IR-dominant,EXCL}",
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to trained KPFuse v2 checkpoint (.pth)")
    p.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional dataset folder names (default: auto-discover all folders with vi/ir)",
    )
    p.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto (default)")
    p.add_argument("--pad-multiple", type=int, default=16)
    p.add_argument("--save-ext", type=str, default="", help="Force output extension, e.g. .png")
    p.add_argument("--strict", dest="strict", action="store_true")
    p.add_argument("--no-strict", dest="strict", action="store_false")
    p.set_defaults(strict=True)

    # Architecture/smoothing args for checkpoint compatibility with kpfuse_v2.py
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
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    datasets_root = Path(args.datasets_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"
    autocast_dev = "cuda" if use_amp else "cpu"

    model = FusionNet(
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

    state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=args.strict)
    model.eval()

    detector = KeyNetResponse().to(device).eval()
    dataset_dirs = discover_datasets(datasets_root, args.datasets)
    if not dataset_dirs:
        raise RuntimeError(f"No dataset folders with vi/ir found under: {datasets_root}")

    total_saved = 0
    for ds_dir in dataset_dirs:
        vi_dir = ds_dir / "vi"
        ir_dir = ds_dir / "ir"

        vis_map = build_stem_map(list_images(vi_dir))
        ir_map = build_stem_map(list_images(ir_dir))
        common_stems = sorted(set(vis_map.keys()) & set(ir_map.keys()))
        if not common_stems:
            print(f"[skip] {ds_dir.name}: no matched vi/ir pairs")
            continue

        out_vis = output_root / ds_dir.name / "VIS-dominant"
        out_ir = output_root / ds_dir.name / "IR-dominant"
        out_excl = output_root / ds_dir.name / "EXCL"
        out_vis.mkdir(parents=True, exist_ok=True)
        out_ir.mkdir(parents=True, exist_ok=True)
        out_excl.mkdir(parents=True, exist_ok=True)

        print(f"[dataset] {ds_dir.name}: {len(common_stems)} pairs")
        for idx, stem in enumerate(common_stems, 1):
            vis_path = vis_map[stem]
            ir_path = ir_map[stem]
            v, i = load_pair(vis_path, ir_path)
            v = v.unsqueeze(0).to(device)
            i = i.unsqueeze(0).to(device)
            v, pad_hw = pad_to_multiple(v, multiple=args.pad_multiple)
            i, _ = pad_to_multiple(i, multiple=args.pad_multiple)

            with torch.amp.autocast(autocast_dev, enabled=use_amp):
                vis_kmap, ir_kmap = compute_keypoint_maps(v, i, detector)
                fused_vis, fused_ir, fused_excl = model(
                    v, i, vis_kmap=vis_kmap, ir_kmap=ir_kmap, return_all=True
                )

            fused_vis = unpad(fused_vis, pad_hw).squeeze(0).detach().cpu().clamp_(0.0, 1.0)
            fused_ir = unpad(fused_ir, pad_hw).squeeze(0).detach().cpu().clamp_(0.0, 1.0)
            fused_excl = unpad(fused_excl, pad_hw).squeeze(0).detach().cpu().clamp_(0.0, 1.0)

            out_ext = args.save_ext if args.save_ext else vis_path.suffix
            if out_ext and not out_ext.startswith("."):
                out_ext = "." + out_ext
            out_name = f"{vis_path.stem}{out_ext}"

            TF.to_pil_image(fused_vis).save(out_vis / out_name)
            TF.to_pil_image(fused_ir).save(out_ir / out_name)
            TF.to_pil_image(fused_excl).save(out_excl / out_name)

            total_saved += 3
            if idx % 50 == 0 or idx == len(common_stems):
                print(f"  saved {idx}/{len(common_stems)} pairs")

    print(f"[done] total images written: {total_saved}")


if __name__ == "__main__":
    main()
