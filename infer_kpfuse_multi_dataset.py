import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from kpfuse import FusionNet, KeyNetResponse, compute_keypoint_maps


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def build_stem_map(files: Sequence[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in files:
        key = p.stem
        if key not in out:
            out[key] = p
    return out


def discover_datasets(input_root: Path, dataset_names: Sequence[str]) -> List[Path]:
    if dataset_names:
        ds = [input_root / name for name in dataset_names]
    else:
        ds = [p for p in sorted(input_root.iterdir()) if p.is_dir() and (p / "vi").is_dir() and (p / "ir").is_dir()]
    ds = [p for p in ds if (p / "vi").is_dir() and (p / "ir").is_dir()]
    return ds


def load_pair(vis_path: Path, ir_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    with Image.open(vis_path) as v_img, Image.open(ir_path) as i_img:
        v = TF.to_tensor(v_img.convert("RGB"))
        i = TF.to_tensor(i_img.convert("L"))
    return v, i


def pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, Tuple[int, int]]:
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
    p = argparse.ArgumentParser("KPFuse multi-dataset inference")
    p.add_argument("--input-root", type=str, required=True, help="Root containing dataset folders")
    p.add_argument("--output-root", type=str, required=True, help="Root to save fused outputs")
    p.add_argument("--ckpt", type=str, required=True, help="Path to trained KPFuse checkpoint (.pth)")
    p.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional dataset folder names under input-root (default: auto-discover folders with vi/ir)",
    )
    p.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto (default)")
    p.add_argument("--save-ext", type=str, default="", help="Optional forced output extension (e.g. .png).")
    p.add_argument("--strict", dest="strict", action="store_true")
    p.add_argument("--no-strict", dest="strict", action="store_false")
    p.set_defaults(strict=True)
    # Keep architecture flags available in case checkpoint was trained with non-default capacity.
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--bottleneck-ch", type=int, default=256)
    p.add_argument("--attn-heads", type=int, default=16)
    p.add_argument("--attn-sr", type=int, default=4)
    p.add_argument("--attn-depth", type=int, default=3)
    p.add_argument("--attn-mlp-ratio", type=float, default=2.0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    input_root = Path(args.input_root)
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
        attn_sr=args.attn_sr,
        attn_depth=args.attn_depth,
        attn_mlp_ratio=args.attn_mlp_ratio,
    ).to(device)

    state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=args.strict)
    model.eval()

    detector = KeyNetResponse().to(device).eval()

    dataset_dirs = discover_datasets(input_root, args.datasets)
    if not dataset_dirs:
        raise RuntimeError(f"No dataset folders with vi/ir found under: {input_root}")

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

        out_dir = output_root / ds_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[dataset] {ds_dir.name}: {len(common_stems)} matched pairs")
        for idx, stem in enumerate(common_stems, 1):
            vis_path = vis_map[stem]
            ir_path = ir_map[stem]
            v, i = load_pair(vis_path, ir_path)
            v = v.unsqueeze(0).to(device)
            i = i.unsqueeze(0).to(device)

            # Network uses stride-8 pyramid; reflect-pad then unpad output back.
            v, pad_hw = pad_to_multiple(v, multiple=8)
            i, _ = pad_to_multiple(i, multiple=8)

            with torch.amp.autocast(autocast_dev, enabled=use_amp):
                vis_kmap, ir_kmap = compute_keypoint_maps(v, i, detector)
                fused = model(v, i, vis_kmap=vis_kmap, ir_kmap=ir_kmap)

            fused = unpad(fused, pad_hw).squeeze(0).detach().cpu().clamp_(0.0, 1.0)
            out_ext = args.save_ext if args.save_ext else vis_path.suffix
            if out_ext and not out_ext.startswith("."):
                out_ext = "." + out_ext
            out_name = vis_path.stem + out_ext
            out_path = out_dir / out_name

            fused_pil = TF.to_pil_image(fused)
            if out_path.suffix.lower() in {".jpg", ".jpeg"}:
                fused_pil.save(out_path, quality=95)
            else:
                fused_pil.save(out_path)

            total_saved += 1
            if idx % 50 == 0 or idx == len(common_stems):
                print(f"  saved {idx}/{len(common_stems)}")

    print(f"[done] total fused images saved: {total_saved}")


if __name__ == "__main__":
    main()
