import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from pytorch_msssim import ssim
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Fixed-size inputs benefit from benchmark autotuning.
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# -------------------------------------------------------------
# 1. Model Components: Cross-Attention + Spatial Reduction
# -------------------------------------------------------------

class CrossSRA(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x_vis, x_ir, H, W):
        B, N, C = x_vis.shape
        q = self.q(x_vis).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Keys and Values from Infrared (Reduced spatially to save VRAM)
        x_ir_spatial = x_ir.permute(0, 2, 1).reshape(B, C, H, W)
        if self.sr_ratio > 1:
            x_ir_tokens = self.sr(x_ir_spatial).flatten(2).transpose(1, 2)
            x_ir_tokens = self.norm(x_ir_tokens)
        else:
            x_ir_tokens = x_ir_spatial.flatten(2).transpose(1, 2)

        k = self.k(x_ir_tokens).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_ir_tokens).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class SourceFocusFusionNet(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        def block(ic, oc): return nn.Sequential(nn.Conv2d(ic, oc, 3, 2, 1), nn.BatchNorm2d(oc), nn.LeakyReLU(0.2))
        self.v_enc = nn.ModuleList([block(3, 16), block(16, 32), block(32, d_model)])
        self.i_enc = nn.ModuleList([block(1, 16), block(16, 32), block(32, d_model)])
        self.bridge = CrossSRA(d_model, sr_ratio=8)
        self.dec = nn.ModuleList([nn.Conv2d(d_model, 32, 3, 1, 1), nn.Conv2d(32, 16, 3, 1, 1), nn.Conv2d(16, 3, 3, 1, 1)])

    def forward(self, vis, ir):
        v1 = self.v_enc[0](vis); i1 = self.i_enc[0](ir)
        v2 = self.v_enc[1](v1);  i2 = self.i_enc[1](i1)
        v3 = self.v_enc[2](v2);  i3 = self.i_enc[2](i2)
        B, C, H, W = v3.shape
        v_f, i_f = v3.flatten(2).transpose(1, 2), i3.flatten(2).transpose(1, 2)
        f_f = v_f + self.bridge(v_f, i_f, H, W)
        f3 = f_f.transpose(1, 2).reshape(B, C, H, W)
        d3 = F.interpolate(F.relu(self.dec[0](f3)), scale_factor=2, mode="bilinear", align_corners=False)
        d2 = F.interpolate(F.relu(self.dec[1](d3)), scale_factor=2, mode="bilinear", align_corners=False)
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        out = torch.sigmoid(self.dec[2](d1))
        return out, v3, i3, f3

# -------------------------------------------------------------
# 2. Optimized Source-Focus Loss
# -------------------------------------------------------------

class SourceFocusLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel Kernels for Gradient Extraction
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx); self.register_buffer('ky', ky)

    def get_grad(self, x):
        if x.shape[1] == 3: x = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]
        gx = F.conv2d(x, self.kx, padding=1); gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def nuclear_norm(self, x):
        flattened = x.float().reshape(x.size(0), x.size(1), -1)
        return torch.linalg.norm(flattened, ord='nuc', dim=(1, 2)).mean()

    def forward(self, fused, vis, ir, gt, v_feat, i_feat, f_feat):
        # 1. Structural Guide (Low Weight on GT)
        l_ssim = 1 - ssim(fused, gt, data_range=1.0)
        l_mse = F.mse_loss(fused, gt)
        
        # 2. Source-Saliency Detail (High Weight on Raw Sensors)
        # We ignore GT gradients and force the model to match the sharpest sensor source
        g_f = self.get_grad(fused)
        g_v = self.get_grad(vis)
        g_i = self.get_grad(ir)
        g_target = torch.max(g_v, g_i)
        l_grad = F.l1_loss(g_f, g_target)

        # 3. Nuclear Loss (Feature Diversity)
        l_nuc = torch.relu(0.5 * (self.nuclear_norm(v_feat) + self.nuclear_norm(i_feat)) - self.nuclear_norm(f_feat))

        # Final Balance: Prioritize gradients over pixel-averaging
        return (10.0 * l_ssim) + (2.0 * l_mse) + (30.0 * l_grad) + (0.5 * l_nuc)

# -------------------------------------------------------------
# 3. Training Utilities: Early Stopping & Scheduler
# -------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=10, path='best_source_focus.pth'):
        self.patience, self.path, self.counter, self.best_loss, self.early_stop = patience, path, 0, None, False
    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            self.counter = 0; print("⭐ New Best Val Loss. Model Saved.")
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True


class FusionDataset(Dataset):
    def __init__(self, v_path, i_path, g_path, files, clahe_prob=0.5):
        self.v_path = v_path
        self.i_path = i_path
        self.g_path = g_path
        self.files = files
        self.clahe_prob = clahe_prob
        
        # Initialize CLAHE object once
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Standardize transform pipeline
        self.rgb_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.ir_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def apply_clahe(self, pil_img):
        """Applies CLAHE in LAB space to enhance contrast without color distortion."""
        # Convert PIL to Numpy (RGB)
        img_np = np.array(pil_img)
        
        # LAB conversion for luminance-only enhancement
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        l_enhanced = self.clahe.apply(l)
        
        # Merge and convert back to RGB
        enhanced_img = cv2.merge((l_enhanced, a, b))
        final_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(final_rgb)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f_name = self.files[idx]
        
        # Path construction
        v_p = os.path.join(self.v_path, f_name)
        i_p = os.path.join(self.i_path, f_name)
        g_p = os.path.join(self.g_path, f_name)

        # 1. Open files using context managers (ensures they close immediately)
        try:
            with Image.open(v_p) as v_raw, Image.open(i_p) as i_raw, Image.open(g_p) as g_raw:
                # Load images into memory and convert
                vis = v_raw.convert('RGB')
                ir = i_raw.convert('L')
                gt = g_raw.convert('RGB')

                # 2. Random Contrast Augmentation (0.5 probability)
                if random.random() < self.clahe_prob:
                    vis = self.apply_clahe(vis)
                    gt = self.apply_clahe(gt)

                # 3. Transformations
                vis_tensor = self.rgb_transform(vis)
                ir_tensor = self.ir_transform(ir)
                gt_tensor = self.rgb_transform(gt)

                return vis_tensor, ir_tensor, gt_tensor, f_name

        except Exception as e:
            print(f"Error loading {f_name}: {e}")
            # Skip corrupt samples instead of injecting zero-images.
            return None


def collate_skip_none(batch):
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None
    vis, ir, gt, names = zip(*batch)
    return torch.stack(vis), torch.stack(ir), torch.stack(gt), names

# -------------------------------------------------------------
# 4. Main Training Logic
# -------------------------------------------------------------

def main(train_loader, val_loader, model_path, epochs=100, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = SourceFocusFusionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = SourceFocusLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stop = EarlyStopping(patience=12, path=model_path)

    for epoch in range(epochs):
        model.train(); train_l = 0.0; train_steps = 0
        for batch in train_loader:
            if batch is None:
                continue

            vis, ir, gt, _ = batch
            vis = vis.to(device, non_blocking=True)
            ir = ir.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                fused, v_f, i_f, f_f = model(vis, ir)
                # Loss Calculation (Normalization to [0,1] for SSIM/Grad logic)
                loss = criterion(fused, (vis + 1) / 2, (ir + 1) / 2, (gt + 1) / 2, v_f, i_f, f_f)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_l += loss.item()
            train_steps += 1

        if train_steps == 0:
            raise RuntimeError("No valid training batches were produced. Check dataset integrity.")

        model.eval(); val_l = 0.0; val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                vis, ir, gt, _ = batch
                vis = vis.to(device, non_blocking=True)
                ir = ir.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    fused, v_f, i_f, f_f = model(vis, ir)
                    val_loss = criterion(fused, (vis + 1) / 2, (ir + 1) / 2, (gt + 1) / 2, v_f, i_f, f_f)

                val_l += val_loss.item()
                val_steps += 1

        if val_steps == 0:
            raise RuntimeError("No valid validation batches were produced. Check validation split/data.")

        avg_t_loss = train_l / train_steps
        avg_v_loss = val_l / val_steps
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1} | LR: {current_lr:.2e} | Train Loss: {avg_t_loss:.4f} | Val Loss: {avg_v_loss:.4f}")

        scheduler.step(avg_v_loss)
        early_stop(avg_v_loss, model)
        if early_stop.early_stop:
            print("🛑 Early Stopping Triggered.")
            break


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Source-focus VIS/IR fusion training")
    parser.add_argument("--v-dir", default="/home/chandra/Documents/Mizzou/VIS_IR_Fusion/training_data/vi")
    parser.add_argument("--i-dir", default="/home/chandra/Documents/Mizzou/VIS_IR_Fusion/training_data/ir")
    parser.add_argument("--g-dir", default="/home/chandra/Documents/Mizzou/VIS_IR_Fusion/training_data/SGT_Direct_Fusion")
    parser.add_argument("--model-path", default="/home/chandra/Documents/Mizzou/outputs_100epochs/v7_dynamic.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clahe-prob", type=float, default=0.5)
    return parser

def validate_dirs(*paths):
    for path in paths:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}")


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    validate_dirs(args.v_dir, args.i_dir, args.g_dir)

    valid_ext = (".png", ".jpg", ".jpeg")
    files = []
    for f in sorted(os.listdir(args.v_dir)):
        if not f.lower().endswith(valid_ext):
            continue
        i_path = os.path.join(args.i_dir, f)
        g_path = os.path.join(args.g_dir, f)
        if os.path.isfile(i_path) and os.path.isfile(g_path):
            files.append(f)

    if len(files) < 2:
        raise RuntimeError("Need at least 2 matched files across vis/ir/gt folders.")

    random.shuffle(files)
    split = int(args.train_split * len(files))
    split = max(1, min(split, len(files) - 1))
    train_files = files[:split]
    val_files = files[split:]

    print(f"Found {len(files)} matched samples | Train: {len(train_files)} | Val: {len(val_files)}")

    pin_memory = torch.cuda.is_available()
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_skip_none,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        persistent_workers=args.num_workers > 0,
    )
    val_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_skip_none,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2
        val_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        FusionDataset(args.v_dir, args.i_dir, args.g_dir, train_files, args.clahe_prob),
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        FusionDataset(args.v_dir, args.i_dir, args.g_dir, val_files, 0.0),
        **val_loader_kwargs,
    )

    model_dir = os.path.dirname(args.model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print("🚀 Starting Source-Focus training...")
    main(
        train_loader=train_loader,
        val_loader=val_loader,
        model_path=args.model_path,
        epochs=args.epochs,
        lr=args.lr,
    )
    print("✅ Training Finished.")