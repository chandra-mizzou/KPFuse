import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from kpfuse import KeyNetResponse


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def build_stem_map(files: Sequence[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in files:
        if p.stem not in out:
            out[p.stem] = p
    return out


def resolve_child_dir(parent: Path, desired_name: str) -> Path:
    """Return child dir matching desired_name, case-insensitive if needed."""
    direct = parent / desired_name
    if direct.is_dir():
        return direct
    desired_cf = desired_name.casefold()
    for child in parent.iterdir():
        if child.is_dir() and child.name.casefold() == desired_cf:
            return child
    return direct


def discover_dataset_dirs(
    datasets_root: Path,
    dataset_names: Sequence[str],
    vis_dirname: str,
    ir_dirname: str,
) -> List[Tuple[str, Path, Path]]:
    out: List[Tuple[str, Path, Path]] = []
    if dataset_names:
        candidate_dirs = [(name, datasets_root / name) for name in dataset_names]
    else:
        candidate_dirs = [(p.name, p) for p in sorted(datasets_root.iterdir()) if p.is_dir()]

    for ds_name, ds_dir in candidate_dirs:
        vis_dir = resolve_child_dir(ds_dir, vis_dirname)
        ir_dir = resolve_child_dir(ds_dir, ir_dirname)
        if vis_dir.is_dir() and ir_dir.is_dir():
            out.append((ds_name, vis_dir, ir_dir))
    return out


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-6)


def retention_from_responses(
    src_resp: torch.Tensor,
    fused_resp: torch.Tensor,
    topk: int,
    tolerance_px: int,
) -> torch.Tensor:
    if fused_resp.shape[-2:] != src_resp.shape[-2:]:
        fused_resp = F.interpolate(
            fused_resp,
            size=src_resp.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    bsz, ch, h, w = src_resp.shape
    src_flat = src_resp.flatten(1)
    fused_flat = fused_resp.flatten(1)
    k = min(topk, src_flat.shape[1], fused_flat.shape[1])
    if k < 1:
        raise ValueError("topk must be >= 1")

    src_idx = src_flat.topk(k=k, dim=1).indices
    fused_idx = fused_flat.topk(k=k, dim=1).indices

    src_mask = torch.zeros_like(src_flat, dtype=torch.float32)
    fused_mask = torch.zeros_like(fused_flat, dtype=torch.float32)
    src_mask.scatter_(1, src_idx, 1.0)
    fused_mask.scatter_(1, fused_idx, 1.0)

    src_mask = src_mask.view(bsz, ch, h, w)
    fused_mask = fused_mask.view(bsz, ch, h, w)

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


def load_triplet_tensors(
    vis_path: Path,
    ir_path: Path,
    fused_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with Image.open(vis_path) as v_img, Image.open(ir_path) as i_img, Image.open(fused_path) as f_img:
        v = TF.to_tensor(v_img.convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)
        i = TF.to_tensor(i_img.convert("L")).unsqueeze(0).to(device=device, dtype=dtype)
        f = TF.to_tensor(f_img.convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)
    return v, i, f


@torch.no_grad()
def evaluate_dataset_model(
    detector: KeyNetResponse,
    vis_dir: Path,
    ir_dir: Path,
    fused_dir: Path,
    device: torch.device,
    topk: int,
    tolerance_px: int,
) -> Tuple[float, float, float, int]:
    vis_map = build_stem_map(list_images(vis_dir))
    ir_map = build_stem_map(list_images(ir_dir))
    fused_map = build_stem_map(list_images(fused_dir))
    common_stems = sorted(set(vis_map.keys()) & set(ir_map.keys()) & set(fused_map.keys()))
    if not common_stems:
        return float("nan"), float("nan"), float("nan"), 0

    detector_dtype = next(detector.parameters()).dtype
    sum_vis = 0.0
    sum_ir = 0.0

    for stem in common_stems:
        v, i, f = load_triplet_tensors(
            vis_path=vis_map[stem],
            ir_path=ir_map[stem],
            fused_path=fused_map[stem],
            device=device,
            dtype=detector_dtype,
        )
        rv = normalize_map(detector(v.mean(1, True)))
        ri = normalize_map(detector(i))
        rf = normalize_map(detector(f.mean(1, True)))

        vis_ret = retention_from_responses(rv, rf, topk=topk, tolerance_px=tolerance_px)
        ir_ret = retention_from_responses(ri, rf, topk=topk, tolerance_px=tolerance_px)
        sum_vis += vis_ret.item()
        sum_ir += ir_ret.item()

    n = len(common_stems)
    vis_avg = sum_vis / n
    ir_avg = sum_ir / n
    mean_avg = 0.5 * (vis_avg + ir_avg)
    return vis_avg, ir_avg, mean_avg, n


def format_float(x: float) -> str:
    if x != x:  # NaN check
        return "NA"
    return f"{x:.4f}"


def write_table_csv(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    headers = [
        "S No.",
        "Dataset",
        "Model Name",
        "Avg VIS-Fused Retention",
        "Avg IR-Fused Retention",
        "Avg Retention",
        "Matched Pairs",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_table_markdown(md_path: Path, rows: List[Dict[str, str]], dataset_name: str) -> None:
    headers = [
        "S No.",
        "Dataset",
        "Model Name",
        "Avg VIS-Fused Retention",
        "Avg IR-Fused Retention",
        "Avg Retention",
        "Matched Pairs",
    ]
    lines = [f"# {dataset_name}", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join([row[h] for h in headers]) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser("Evaluate keypoint retention for multiple fusion models")
    p.add_argument("--datasets-root", type=str, required=True, help="Root containing dataset1..dataset6 with vi/ir")
    p.add_argument("--models-root", type=str, required=True, help="Root containing model folders (13 fusion models)")
    p.add_argument("--output-dir", type=str, default="retention_tables", help="Directory to save output tables")
    p.add_argument(
        "--dataset-names",
        nargs="*",
        default=[],
        help="Optional dataset folder names under datasets-root (default: auto-discover valid dataset folders)",
    )
    p.add_argument(
        "--model-dataset-names",
        nargs="*",
        default=[],
        help=(
            "Optional model-side dataset folder names under each model folder. "
            "If provided, must align by position with discovered/selected datasets-root names "
            "(example: --dataset-names LLVIP TNO ... --model-dataset-names dataset1 dataset2 ...)."
        ),
    )
    p.add_argument(
        "--vis-dirname",
        type=str,
        default="vi",
        help="VIS subfolder name under each dataset (case-insensitive match supported).",
    )
    p.add_argument(
        "--ir-dirname",
        type=str,
        default="ir",
        help="IR subfolder name under each dataset (case-insensitive match supported).",
    )
    p.add_argument(
        "--fused-subdir",
        type=str,
        default="",
        help="Optional subfolder name under model/dataset containing fused images (e.g. fusedimages)",
    )
    p.add_argument("--topk", type=int, default=180)
    p.add_argument("--tolerance-px", type=int, default=3)
    p.add_argument("--expected-model-count", type=int, default=13, help="Warn if discovered model folders differ")
    p.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto (default)")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    datasets_root = Path(args.datasets_root)
    models_root = Path(args.models_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_dirs = sorted([p for p in models_root.iterdir() if p.is_dir()])
    if args.expected_model_count > 0 and len(model_dirs) != args.expected_model_count:
        print(
            f"[warn] Expected {args.expected_model_count} model folders under {models_root}, "
            f"found {len(model_dirs)}"
        )

    detector = KeyNetResponse().to(device).eval()
    print(f"[info] device={device}, models={len(model_dirs)}")

    dataset_specs = discover_dataset_dirs(
        datasets_root=datasets_root,
        dataset_names=args.dataset_names,
        vis_dirname=args.vis_dirname,
        ir_dirname=args.ir_dirname,
    )
    if not dataset_specs:
        raise RuntimeError(
            f"No valid datasets found under {datasets_root} with subfolders "
            f"{args.vis_dirname!r}/{args.ir_dirname!r}"
        )
    if args.model_dataset_names and len(args.model_dataset_names) != len(dataset_specs):
        raise ValueError(
            "--model-dataset-names must have the same number of items as selected datasets "
            f"({len(dataset_specs)}), got {len(args.model_dataset_names)}"
        )

    for ds_idx, (dataset_name, vis_dir, ir_dir) in enumerate(dataset_specs, 1):
        model_dataset_name = (
            args.model_dataset_names[ds_idx - 1]
            if args.model_dataset_names
            else dataset_name
        )

        rows: List[Dict[str, str]] = []
        print(f"[dataset] {dataset_name} (model folder: {model_dataset_name})")

        for idx, model_dir in enumerate(model_dirs, 1):
            fused_dir = model_dir / model_dataset_name
            if args.fused_subdir:
                fused_dir = fused_dir / args.fused_subdir

            if not fused_dir.is_dir():
                vis_avg = float("nan")
                ir_avg = float("nan")
                mean_avg = float("nan")
                matched = 0
            else:
                vis_avg, ir_avg, mean_avg, matched = evaluate_dataset_model(
                    detector=detector,
                    vis_dir=vis_dir,
                    ir_dir=ir_dir,
                    fused_dir=fused_dir,
                    device=device,
                    topk=args.topk,
                    tolerance_px=args.tolerance_px,
                )

            row = {
                "S No.": str(idx),
                "Dataset": dataset_name,
                "Model Name": model_dir.name,
                "Avg VIS-Fused Retention": format_float(vis_avg),
                "Avg IR-Fused Retention": format_float(ir_avg),
                "Avg Retention": format_float(mean_avg),
                "Matched Pairs": str(matched),
            }
            rows.append(row)
            print(
                f"  {idx:02d}. {model_dir.name} | "
                f"vis={row['Avg VIS-Fused Retention']} "
                f"ir={row['Avg IR-Fused Retention']} "
                f"mean={row['Avg Retention']} "
                f"pairs={matched}"
            )

        csv_path = out_root / f"{dataset_name}_retention_table.csv"
        md_path = out_root / f"{dataset_name}_retention_table.md"
        write_table_csv(csv_path, rows)
        write_table_markdown(md_path, rows, dataset_name)
        print(f"  [saved] {csv_path}")
        print(f"  [saved] {md_path}")

    print(f"[done] Tables saved under: {out_root}")


if __name__ == "__main__":
    main()
