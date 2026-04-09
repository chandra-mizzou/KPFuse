import argparse
import os
import random
from pathlib import Path
from typing import Optional, Sequence

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

    def forward(self, qx, kvx, h, w):
        bsz, n, c = qx.shape
        q = self.q(qx).reshape(bsz, n, self.h, c // self.h).permute(0, 2, 1, 3)

        kv = kvx.permute(0, 2, 1).reshape(bsz, c, h, w)
        if self.sr > 1:
            kv = self.conv(kv).flatten(2).transpose(1, 2)
            kv = self.norm(kv)
        else:
            kv = kv.flatten(2).transpose(1, 2)

        k = self.k(kv).reshape(bsz, -1, self.h, c // self.h).permute(0, 2, 1, 3)
        v = self.v(kv).reshape(bsz, -1, self.h, c // self.h).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
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

    def forward(self, vf, if_, h, w):
        attn = torch.sigmoid(self.a) * self.v2i(vf, if_, h, w)
        attn = attn + torch.sigmoid(self.b) * self.i2v(if_, vf, h, w)
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

    def forward(self, v, i):
        v1 = self.v1(v)
        v2 = self.v2(v1)
        v3 = self.v3(v2)

        i1 = self.i1(i)
        i2 = self.i2(i1)
        i3 = self.i3(i2)

        bsz, c, h, w = v3.shape
        vf = v3.flatten(2).transpose(1, 2)
        if_ = i3.flatten(2).transpose(1, 2)

        f = vf
        for block in self.fusion_blocks:
            f = block(f, if_, h, w)
        f = f.transpose(1, 2).reshape(bsz, c, h, w)

        d3 = self.c3(torch.cat([self.up3(f), v2, i2], dim=1))
        d2 = self.c2(torch.cat([self.up2(d3), v1, i1], dim=1))
        d1 = self.up1(d2)
        return torch.sigmoid(self.out(d1))


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
    def __init__(self):
        super().__init__()
        self.detector = KeyNetResponse()

    def forward(self, v, i, f):
        v = v.mean(1, True)
        f = f.mean(1, True)

        with torch.no_grad():
            rv = self.detector(v)
            ri = self.detector(i)
        rf = self.detector(f)

        if rf.shape[-2:] != rv.shape[-2:]:
            rf = F.interpolate(rf, size=rv.shape[-2:], mode="bilinear", align_corners=False)
        target = torch.max(rv, ri)
        return F.l1_loss(rf, target)


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
    ):
        super().__init__()
        self.sp = KeyNetLoss()
        self.sobel = SobelGrad()
        self.w_gt_l1 = w_gt_l1
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.w_src_l1 = w_src_l1
        self.w_sp = w_sp

    def forward(self, v, i, gt, f):
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

        sp_term = self.sp(v, i, f)
        return (
            self.w_gt_l1 * gt_l1
            + self.w_ssim * ssim_term
            + self.w_grad * grad_term
            + self.w_src_l1 * src_l1
            + self.w_sp * sp_term
        )


# =========================================================
# TRAINING
# =========================================================
def make_loaders(
    vis_dir,
    ir_dir,
    gt_dir,
    size,
    batch_size,
    workers,
    split,
    seed,
    train_aug=True,
):
    samples = match_triplets(vis_dir, ir_dir, gt_dir)
    if len(samples) < 2:
        raise RuntimeError(
            "Need at least 2 matched triplets with the same filenames in vis/ir/gt."
        )

    n_train = max(1, int(split * len(samples)))
    n_val = len(samples) - n_train
    if n_val == 0:
        n_train -= 1
        n_val = 1

    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(samples), generator=gen).tolist()
    tr_ids = order[:n_train]
    vl_ids = order[n_train:]
    tr_ds = FusionDataset(samples, size=size, augment=train_aug, indices=tr_ids)
    vl_ds = FusionDataset(samples, size=size, augment=False, indices=vl_ids)

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
    return tr_loader, vl_loader, len(samples)


def make_test_loader(vis_dir, ir_dir, size, batch_size, workers):
    ds = PairDataset(vis_dir, ir_dir, size=size)
    pin = torch.cuda.is_available()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
    )
    return loader, len(ds)


@torch.no_grad()
def save_epoch_previews(
    net,
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

    net.eval()
    saved = 0
    for batch in loader:
        if include_gt:
            v, i, gt = batch
            names = None
        else:
            v, i, names = batch
            gt = None

        v = v.to(device, non_blocking=True)
        i = i.to(device, non_blocking=True)
        with torch.amp.autocast(autocast_dev, enabled=use_amp):
            f = net(v, i)

        v = v.detach().cpu()
        i = i.detach().cpu()
        f = f.detach().cpu().clamp_(0.0, 1.0)
        if gt is not None:
            gt = gt.detach().cpu()

        bsz = f.shape[0]
        for bi in range(bsz):
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

    print(f"Saved {saved} preview sample(s) to: {epoch_dir}")


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    autocast_dev = "cuda" if use_amp else "cpu"

    tr_loader, vl_loader, n_samples = make_loaders(
        args.vis_dir,
        args.ir_dir,
        args.gt_dir,
        args.size,
        args.batch_size,
        args.workers,
        args.split,
        args.seed,
        args.train_aug,
    )
    print(f"Matched samples: {n_samples} | train batches: {len(tr_loader)} | val batches: {len(vl_loader)}")

    preview_loader = vl_loader
    preview_with_gt = True
    if args.test_vis_dir and args.test_ir_dir:
        preview_loader, n_test = make_test_loader(
            args.test_vis_dir,
            args.test_ir_dir,
            args.size,
            args.test_batch_size,
            args.workers,
        )
        preview_with_gt = False
        print(f"Preview source: test pairs | samples: {n_test}")
    else:
        print("Preview source: validation split (set --test-vis-dir/--test-ir-dir to use a separate test folder)")

    net = FusionNet(
        base_ch=args.base_ch,
        bottleneck_ch=args.bottleneck_ch,
        attn_heads=args.attn_heads,
        attn_sr=args.attn_sr,
        attn_depth=args.attn_depth,
        attn_mlp_ratio=args.attn_mlp_ratio,
    ).to(device)
    loss_fn = Loss(
        w_gt_l1=args.w_gt_l1,
        w_ssim=args.w_ssim,
        w_grad=args.w_grad,
        w_src_l1=args.w_src_l1,
        w_sp=args.w_sp,
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


    for e in range(args.epochs):
        print(f"Epoch {e + 1}/{args.epochs}")
        net.train()
        tl = 0.0
        for v, i, gt in tr_loader:
            v = v.to(device, non_blocking=True)
            i = i.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(autocast_dev, enabled=use_amp):
                f = net(v, i)
                loss = loss_fn(v, i, gt, f)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            tl += loss.item()

        net.eval()
        vloss = 0.0
        with torch.no_grad():
            for v, i, gt in vl_loader:
                v = v.to(device, non_blocking=True)
                i = i.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                with torch.amp.autocast(autocast_dev, enabled=use_amp):
                    vloss += loss_fn(v, i, gt, net(v, i)).item()

        tl /= max(1, len(tr_loader))
        vloss /= max(1, len(vl_loader))
        scheduler.step(vloss)
        print(f"TL={tl:.4f} VL={vloss:.4f} LR={opt.param_groups[0]['lr']:.2e}")

        if vloss < best:
            best = vloss
            cnt = 0
            torch.save(net.state_dict(), args.ckpt)
            print(f"Saved best model to: {args.ckpt}")
        else:
            cnt += 1

        save_epoch_previews(
            net=net,
            loader=preview_loader,
            device=device,
            use_amp=use_amp,
            out_root=args.preview_dir,
            epoch=e + 1,
            max_items=args.preview_count,
            include_gt=preview_with_gt,
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
    p.add_argument("--split", type=float, default=0.8)
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
    p.add_argument("--test-vis-dir", type=str, default="", help="Optional VIS folder for test-time per-epoch previews")
    p.add_argument("--test-ir-dir", type=str, default="", help="Optional IR folder for test-time per-epoch previews")
    p.add_argument("--test-batch-size", type=int, default=4)
    p.add_argument("--train-aug", action=argparse.BooleanOptionalAction, default=True)
    # Rebalanced defaults after switching SSIM/gradient supervision to GT.
    p.add_argument("--w-gt-l1", type=float, default=1.0)
    p.add_argument("--w-ssim", type=float, default=0.9)
    p.add_argument("--w-grad", type=float, default=0.7)
    p.add_argument("--w-src-l1", type=float, default=0.25)
    p.add_argument("--w-sp", type=float, default=0.08)
    # Capacity scaling knobs for higher GPU utilization.
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--bottleneck-ch", type=int, default=256)
    p.add_argument("--attn-heads", type=int, default=16)
    p.add_argument("--attn-sr", type=int, default=4)
    p.add_argument("--attn-depth", type=int, default=3)
    p.add_argument("--attn-mlp-ratio", type=float, default=2.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if bool(args.test_vis_dir) ^ bool(args.test_ir_dir):
        raise ValueError("Set both --test-vis-dir and --test-ir-dir, or leave both unset.")
    for req_path in [args.vis_dir, args.ir_dir, args.gt_dir]:
        if not os.path.isdir(req_path):
            raise FileNotFoundError(f"Missing directory: {req_path}")
    for opt_path in [args.test_vis_dir, args.test_ir_dir]:
        if opt_path and not os.path.isdir(opt_path):
            raise FileNotFoundError(f"Missing directory: {opt_path}")
    train(args)

# nohup python3 -u bidirectional_crossattn.py --vis-dir '/home/chandra/Documents/Mizzou/VIS_IR_Fusion/training_data/vi' --ir-dir '/home/chandra/Documents/Mizzou/VIS_IR_Fusion/training_data/ir' --gt-dir '/home/chandra/Documents/Mizzou/VIS_IR_Fusion/training_data/SGT_Direct_Fusion' --epochs 100 --batch-size 12 --size 512 --base-ch 48 --bottleneck-ch 192 --attn-depth 3 --attn-heads 8 --attn-sr 2  --workers 4 --ckpt kpfuse_v1.pth  > train_log_kpfuse.txt 2>&1 &
