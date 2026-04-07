import os, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import kornia as K
import kornia.feature as KF

# =========================================================
# DATASET
# =========================================================
def get_all_images(folder):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

class FusionDataset(Dataset):
    def __init__(self, vis_dir, ir_dir, gt_dir, size=256):
        self.vis = get_all_images(vis_dir)
        self.ir = get_all_images(ir_dir)
        self.gt = get_all_images(gt_dir)

        self.tf = T.Compose([
            T.Resize((size, size)),
            T.ToTensor()
        ])

    def __len__(self): return len(self.vis)

    def __getitem__(self, i):
        v = self.tf(Image.open(self.vis[i]).convert("RGB"))
        ir = self.tf(Image.open(self.ir[i]).convert("L"))
        gt = self.tf(Image.open(self.gt[i]).convert("RGB"))
        return v, ir, gt

# =========================================================
# MODEL
# =========================================================
class CrossSRA(nn.Module):
    def __init__(self, dim, heads=8, sr=8):
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

    def forward(self, qx, kvx, H, W):
        B,N,C = qx.shape

        q = self.q(qx).reshape(B,N,self.h,C//self.h).permute(0,2,1,3)

        kv = kvx.permute(0,2,1).reshape(B,C,H,W)
        if self.sr>1:
            kv = self.conv(kv).flatten(2).transpose(1,2)
            kv = self.norm(kv)
        else:
            kv = kv.flatten(2).transpose(1,2)

        k = self.k(kv).reshape(B,-1,self.h,C//self.h).permute(0,2,1,3)
        v = self.v(kv).reshape(B,-1,self.h,C//self.h).permute(0,2,1,3)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(-1)

        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        return self.proj(x)

def down(ic,oc):
    return nn.Sequential(nn.Conv2d(ic,oc,3,2,1),nn.BatchNorm2d(oc),nn.LeakyReLU(0.2))

def up(ic,oc):
    return nn.Sequential(nn.ConvTranspose2d(ic,oc,2,2),nn.BatchNorm2d(oc),nn.LeakyReLU(0.2))

def conv(ic,oc):
    return nn.Sequential(nn.Conv2d(ic,oc,3,1,1),nn.BatchNorm2d(oc),nn.LeakyReLU(0.2))

class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.v1,self.v2,self.v3 = down(3,16),down(16,32),down(32,64)
        self.i1,self.i2,self.i3 = down(1,16),down(16,32),down(32,64)

        self.v2i = CrossSRA(64)
        self.i2v = CrossSRA(64)

        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))

        self.up3 = up(64,32)
        self.c3 = conv(32+32+32,32)
        self.up2 = up(32,16)
        self.c2 = conv(16+16+16,16)
        self.up1 = up(16,16)
        self.out = nn.Conv2d(16,3,3,1,1)

    def forward(self,v,i):
        v1,v2,v3 = self.v1(v),self.v2(self.v1(v)),self.v3(self.v2(self.v1(v)))
        i1,i2,i3 = self.i1(i),self.i2(self.i1(i)),self.i3(self.i2(self.i1(i)))

        B,C,H,W = v3.shape
        vf = v3.flatten(2).transpose(1,2)
        if_ = i3.flatten(2).transpose(1,2)

        f = vf + torch.sigmoid(self.a)*self.v2i(vf,if_,H,W) + torch.sigmoid(self.b)*self.i2v(if_,vf,H,W)
        f = f.transpose(1,2).reshape(B,C,H,W)

        d3 = self.c3(torch.cat([self.up3(f),v2,i2],1))
        d2 = self.c2(torch.cat([self.up2(d3),v1,i1],1))
        d1 = self.up1(d2)

        return torch.sigmoid(self.out(d1))

# =========================================================
# SUPERPOINT + MATCHING LOSS
# =========================================================
class SuperPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = KF.SuperPoint(pretrained=True)
        for p in self.sp.parameters(): p.requires_grad=False

    def forward(self,x):
        o = self.sp(x)
        return o["scores"], o["descriptors"]

class MatchLoss(nn.Module):
    def forward(self,a,b,c):
        a,b,c = [F.normalize(x,1) for x in [a,b,c]]
        sim_ab = torch.einsum("bchw,bcij->bhwij",a,b)
        sim_ac = torch.einsum("bchw,bcij->bhwij",a,c)
        return F.l1_loss(sim_ac,sim_ab)

class SPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = SuperPoint()
        self.match = MatchLoss()

    def forward(self,v,i,f):
        v,i,f = v.mean(1,True),i,f.mean(1,True)
        Sv,Dv = self.sp(v)
        Si,Di = self.sp(i)
        Sf,Df = self.sp(f)

        Sf = F.interpolate(Sf, Sv.shape[-2:])
        Df = F.interpolate(Df, Dv.shape[-2:])

        kp = F.l1_loss(Sf, torch.max(Sv,Si))
        desc = F.l1_loss(Df,(Dv+Di)/2)
        match = self.match(Dv,Di,Df)

        return kp + desc + match

# =========================================================
# OTHER LOSSES
# =========================================================
def grad(x):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],device=x.device).float().view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],device=x.device).float().view(1,1,3,3)
    return torch.sqrt(F.conv2d(x,kx,1,1)**2 + F.conv2d(x,ky,1,1)**2 + 1e-6)

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = SPLoss()

    def forward(self,v,i,f):
        t = torch.max(v,i.repeat(1,3,1,1))
        return (
            F.l1_loss(f,t)
            + 0.5*(1-K.losses.ssim_loss(f,t,11))
            + F.l1_loss(grad(f.mean(1,True)), torch.max(grad(v.mean(1,True)),grad(i)))
            + 2*self.sp(v,i,f)
        )

# =========================================================
# TRAINING
# =========================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = FusionDataset("dataset/vis","dataset/ir","dataset/fused_gt")
    tr,vl = torch.utils.data.random_split(ds,[int(.8*len(ds)),len(ds)-int(.8*len(ds))])

    tr = DataLoader(tr,8,True)
    vl = DataLoader(vl,8)

    net = FusionNet().to(device)
    opt = torch.optim.Adam(net.parameters(),1e-4)
    loss_fn = Loss()

    best,pat,cnt = 1e9,10,0

    for e in range(100):
        print("Epoch Start", (e+1))
        net.train()
        tl=0
        for v,i,_ in tqdm(tr):
            v,i=v.to(device),i.to(device)
            f=net(v,i)
            loss=loss_fn(v,i,f)
            opt.zero_grad(); loss.backward(); opt.step()
            tl+=loss.item()

        net.eval()
        vloss=0
        with torch.no_grad():
            for v,i,_ in vl:
                v,i=v.to(device),i.to(device)
                vloss+=loss_fn(v,i,net(v,i)).item()

        tl/=len(tr); vloss/=len(vl)
        print(f"Epoch {e+1} TL={tl:.4f} VL={vloss:.4f}")

        if vloss<best:
            best=vloss; cnt=0
            torch.save(net.state_dict(),"best.pth")
        else:
            cnt+=1
            if cnt>=pat:
                print("Early stopping")
                break

if __name__=="__main__":
    train()