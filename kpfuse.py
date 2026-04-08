import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

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
    def __init__(self, vis_dir: str, ir_dir: str, gt_dir: str, size: int = 256):
        self.samples = match_triplets(vis_dir, ir_dir, gt_dir)
        if len(self.samples) < 2:
            raise RuntimeError(
                "Need at least 2 matched triplets with the same filenames in vis/ir/gt."
            )

        self.tf = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        vp, ip, gp = self.samples[i]
        with Image.open(vp) as v_img, Image.open(ip) as i_img, Image.open(gp) as g_img:
            v = self.tf(v_img.convert("RGB"))
            ir = self.tf(i_img.convert("L"))
            gt = self.tf(g_img.convert("RGB"))
        return v, ir, gt


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
    def __init__(self):
        super().__init__()
        self.v1, self.v2, self.v3 = down(3, 16), down(16, 32), down(32, 64)
        self.i1, self.i2, self.i3 = down(1, 16), down(16, 32), down(32, 64)

        self.v2i = CrossSRA(64)
        self.i2v = CrossSRA(64)

        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))

        self.up3 = up(64, 32)
        self.c3 = conv(32 + 32 + 32, 32)
        self.up2 = up(32, 16)
        self.c2 = conv(16 + 16 + 16, 16)
        self.up1 = up(16, 16)
        self.out = nn.Conv2d(16, 3, 3, 1, 1)

    def forward(self, v, i):
        # Correct sequential feature extraction (avoid recomputing blocks).
        v1 = self.v1(v)
        v2 = self.v2(v1)
        v3 = self.v3(v2)

        i1 = self.i1(i)
        i2 = self.i2(i1)
        i3 = self.i3(i2)

        bsz, c, h, w = v3.shape
        vf = v3.flatten(2).transpose(1, 2)
        if_ = i3.flatten(2).transpose(1, 2)

        f = (
            vf
            + torch.sigmoid(self.a) * self.v2i(vf, if_, h, w)
            + torch.sigmoid(self.b) * self.i2v(if_, vf, h, w)
        )
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
            # Fallback keeps script trainable when pretrained download is unavailable.
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
        # Use grayscale responses for VIS/Fused; IR is already single-channel.
        v = v.mean(1, True)
        f = f.mean(1, True)

        # Source responses as fixed targets; fused response remains differentiable.
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
def make_loaders(vis_dir, ir_dir, gt_dir, size, batch_size, workers, split, seed):
    ds = FusionDataset(vis_dir, ir_dir, gt_dir, size=size)
    n_train = max(1, int(split * len(ds)))
    n_val = len(ds) - n_train
    if n_val == 0:
        n_train -= 1
        n_val = 1

    gen = torch.Generator().manual_seed(seed)
    tr_ds, vl_ds = random_split(ds, [n_train, n_val], generator=gen)

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
    return tr_loader, vl_loader, len(ds)


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    tr_loader, vl_loader, n_samples = make_loaders(
        args.vis_dir,
        args.ir_dir,
        args.gt_dir,
        args.size,
        args.batch_size,
        args.workers,
        args.split,
        args.seed,
    )
    print(f"Matched samples: {n_samples} | train batches: {len(tr_loader)} | val batches: {len(vl_loader)}")

    net = FusionNet().to(device)
    loss_fn = Loss(
        w_gt_l1=args.w_gt_l1,
        w_ssim=args.w_ssim,
        w_grad=args.w_grad,
        w_src_l1=args.w_src_l1,
        w_sp=args.w_sp,
    ).to(device)
    loss_fn.sp.eval()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = float("inf")
    cnt = 0
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    for e in range(args.epochs):
        print(f"Epoch {e + 1}/{args.epochs}")
        net.train()
        tl = 0.0
        for v, i, gt in tqdm(tr_loader):
            v = v.to(device, non_blocking=True)
            i = i.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                f = net(v, i)
                loss = loss_fn(v, i, gt, f)

            scaler.scale(loss).backward()
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
                with torch.cuda.amp.autocast(enabled=use_amp):
                    vloss += loss_fn(v, i, gt, net(v, i)).item()

        tl /= max(1, len(tr_loader))
        vloss /= max(1, len(vl_loader))
        print(f"TL={tl:.4f} VL={vloss:.4f}")

        if vloss < best:
            best = vloss
            cnt = 0
            torch.save(net.state_dict(), args.ckpt)
            print(f"Saved best model to: {args.ckpt}")
        else:
            cnt += 1
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
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", type=str, default="best_kpfuse.pth")
    # Rebalanced defaults after switching SSIM/gradient supervision to GT.
    p.add_argument("--w-gt-l1", type=float, default=1.0)
    p.add_argument("--w-ssim", type=float, default=0.9)
    p.add_argument("--w-grad", type=float, default=0.7)
    p.add_argument("--w-src-l1", type=float, default=0.25)
    p.add_argument("--w-sp", type=float, default=0.08)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for req_path in [args.vis_dir, args.ir_dir, args.gt_dir]:
        if not os.path.isdir(req_path):
            raise FileNotFoundError(f"Missing directory: {req_path}")
    train(args)
