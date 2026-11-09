import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
import math

# ============================================================
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['TORCH_HOME'] = os.path.join(BASE_DIR, "pretrained")

HR_PATH = os.getenv("MEEREIS_TRAIN_HR", os.path.join(BASE_DIR, "..", "datasets", "10km", "train_HR"))
LR_PATH = os.getenv("MEEREIS_TRAIN_LR", os.path.join(BASE_DIR, "..", "datasets", "10km", "train_LR"))
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints_v4")
LOG_FILE = os.path.join(BASE_DIR, "train_log.txt")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 2. 模型定义 (完整包含 CBAM, ResidualBlock, Generator, Discriminator)
# ============================================================

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.ca(x) * x
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        spatial = self.sa(torch.cat([avg_out, max_out], dim=1))
        return spatial * out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.BatchNorm2d(channels), nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1), nn.BatchNorm2d(channels),
            CBAM(channels)
        )
    def forward(self, x): return x + self.net(x)

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (up_scale**2), 3, 1, 1),
            nn.PixelShuffle(up_scale), nn.PReLU()
        )
    def forward(self, x): return self.net(x)

class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 64, 9, 1, 4), nn.PReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(6)])
        self.block7 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.upsample = nn.Sequential(*[UpsampleBLock(64, 2) for _ in range(int(math.log2(scale_factor)))])
        self.block8 = nn.Conv2d(64, 1, 9, 1, 4)
    def forward(self, x):
        b1 = self.block1(x)
        res = self.res_blocks(b1)
        b7 = self.block7(res)
        out = self.block8(self.upsample(b1 + b7))
        return (torch.tanh(out) + 1) / 2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(128, 256, 1), nn.LeakyReLU(0.2), nn.Conv2d(256, 1, 1)
        )
    def forward(self, x): return self.net(x).view(x.size(0))

# ============================================================
# 3. 数据集与损失函数 (保留老项目效果好的 Loss 比例)
# ============================================================

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir, self.lr_dir = hr_dir, lr_dir
        self.filenames = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png')])
        self.to_tensor = transforms.ToTensor()
    def __len__(self): return len(self.filenames)
    def __getitem__(self, idx):
        name = self.filenames[idx]
        hr = self.to_tensor(Image.open(os.path.join(self.hr_dir, name)).convert('L'))
        lr = self.to_tensor(Image.open(os.path.join(self.lr_dir, name)).convert('L'))
        return lr, hr

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 离线读取 VGG
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:35].eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, gen, target):
        g3, t3 = torch.cat([gen]*3, 1), torch.cat([target]*3, 1)
        g3 = (g3 - self.mean) / self.std
        t3 = (t3 - self.mean) / self.std
        return torch.mean((self.vgg(g3) - self.vgg(t3))**2)

# ============================================================
# ============================================================

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, D = Generator(4).to(device), Discriminator().to(device)
    vgg_l, mse_l, bce_l = VGGPerceptualLoss().to(device), nn.MSELoss(), nn.BCEWithLogitsLoss()
    
    # 采用老项目调优后的学习率
    opt_G = optim.Adam(G.parameters(), lr=1e-4)
    opt_D = optim.Adam(D.parameters(), lr=5e-5)

    start_epoch = 0
    RESUME_PATH = os.path.join(SAVE_DIR, "latest_checkpoint.pth")
    if os.path.exists(RESUME_PATH):
        ckpt = torch.load(RESUME_PATH, map_location=device)
        G.load_state_dict(ckpt['G']); D.load_state_dict(ckpt['D'])
        opt_G.load_state_dict(ckpt['opt_G']); opt_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt['epoch']
        print(f"-> Resumed from epoch {start_epoch}")

    dataset = SRDataset(HR_PATH, LR_PATH)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(start_epoch, 200):
        for i, (lr_imgs, hr_imgs) in enumerate(loader):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # --- 训练 G ---
            opt_G.zero_grad()
            fake_hr = G(lr_imgs)
            l_mse = mse_l(fake_hr, hr_imgs)
            l_vgg = vgg_l(fake_hr, hr_imgs)
            l_adv = bce_l(D(fake_hr), torch.ones_like(D(fake_hr)))
            # 采用老项目验证过的 Loss 权重比例
            loss_g = 1.0 * l_mse + 0.006 * l_vgg + 1e-3 * l_adv
            loss_g.backward(); opt_G.step()

            d_fake_score = torch.sigmoid(D(fake_hr.detach())).mean().item()
            if d_fake_score > 0.1: # 只有生成图让 D 产生疑惑时才练 D，防止 D 过强拉坏结构
                opt_D.zero_grad()
                loss_d = (bce_l(D(hr_imgs), torch.ones_like(D(hr_imgs))) + 
                          bce_l(D(fake_hr.detach()), torch.zeros_like(D(fake_hr.detach())))) / 2
                loss_d.backward(); opt_D.step()

            if i % 20 == 0:
                log_info = f"E[{epoch+1}] B[{i}] G_Loss:{loss_g.item():.4f} D_score:{d_fake_score:.4f}"
                print(log_info)
                with open(LOG_FILE, "a") as f: f.write(log_info + "\n")

        # 保存断点和定期模型
        torch.save({'epoch':epoch+1,'G':G.state_dict(),'D':D.state_dict(),
                    'opt_G':opt_G.state_dict(),'opt_D':opt_D.state_dict()}, RESUME_PATH)
        if (epoch + 1) % 20 == 0:
            torch.save(G.state_dict(), os.path.join(SAVE_DIR, f"G_v4_epoch_{epoch+1}.pth"))
