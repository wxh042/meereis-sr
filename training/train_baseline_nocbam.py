import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
import math

# ============================================================
# 1. 环境与路径配置 (注意：这里修改了保存目录，防止覆盖 V4)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['TORCH_HOME'] = os.path.join(BASE_DIR, "pretrained")

HR_PATH = os.getenv("MEEREIS_TRAIN_HR", os.path.join(BASE_DIR, "..", "datasets", "10km", "train_HR"))
LR_PATH = os.getenv("MEEREIS_TRAIN_LR", os.path.join(BASE_DIR, "..", "datasets", "10km", "train_LR"))

# 修改保存路径为 baseline
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints_baseline")
LOG_FILE = os.path.join(BASE_DIR, "train_log_baseline.txt")
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. 模型定义 (消融：移除 CBAM)
# ============================================================

# 注意：为了结构兼容，这里保留 CBAM 定义但不使用，或者直接去掉。
# 我们在 ResidualBlock 中去掉它。

class ResidualBlock_NoCBAM(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock_NoCBAM, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
            # ⭐ 消融实验：这里去掉了 CBAM(channels)
        )

    def forward(self, x):
        return x + self.net(x)


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (up_scale ** 2), 3, 1, 1),
            nn.PixelShuffle(up_scale), nn.PReLU()
        )

    def forward(self, x): return self.net(x)


class Generator_Baseline(nn.Module):
    def __init__(self, scale_factor):
        super(Generator_Baseline, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 64, 9, 1, 4), nn.PReLU())
        # 使用不含 CBAM 的残差块
        self.res_blocks = nn.Sequential(*[ResidualBlock_NoCBAM(64) for _ in range(6)])
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
# 3. 辅助类 (与原版完全一致，保证实验公平)
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
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:35].eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, gen, target):
        g3, t3 = torch.cat([gen] * 3, 1), torch.cat([target] * 3, 1)
        g3 = (g3 - self.mean) / self.std
        t3 = (t3 - self.mean) / self.std
        return torch.mean((self.vgg(g3) - self.vgg(t3)) ** 2)


# ============================================================
# 4. 训练逻辑 (单一变量：仅模型结构不同)
# ============================================================

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用不含 CBAM 的生成器
    G, D = Generator_Baseline(4).to(device), Discriminator().to(device)
    vgg_l, mse_l, bce_l = VGGPerceptualLoss().to(device), nn.MSELoss(), nn.BCEWithLogitsLoss()

    opt_G = optim.Adam(G.parameters(), lr=1e-4)
    opt_D = optim.Adam(D.parameters(), lr=5e-5)

    dataset = SRDataset(HR_PATH, LR_PATH)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    print("开始消融实验基准训练 (无 CBAM)...")

    # 为了消融对比，建议跑 100 轮左右即可看到明显差异
    for epoch in range(100):
        for i, (lr_imgs, hr_imgs) in enumerate(loader):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # 训练 G
            opt_G.zero_grad()
            fake_hr = G(lr_imgs)
            l_mse = mse_l(fake_hr, hr_imgs)
            l_vgg = vgg_l(fake_hr, hr_imgs)
            l_adv = bce_l(D(fake_hr), torch.ones_like(D(fake_hr)))
            loss_g = 1.0 * l_mse + 0.006 * l_vgg + 1e-3 * l_adv
            loss_g.backward();
            opt_G.step()

            # 训练 D
            d_fake_score = torch.sigmoid(D(fake_hr.detach())).mean().item()
            if d_fake_score > 0.1:
                opt_D.zero_grad()
                loss_d = (bce_l(D(hr_imgs), torch.ones_like(D(hr_imgs))) +
                          bce_l(D(fake_hr.detach()), torch.zeros_like(D(fake_hr.detach())))) / 2
                loss_d.backward();
                opt_D.step()

            if i % 20 == 0:
                log_info = f"E[{epoch + 1}] B[{i}] G_Loss:{loss_g.item():.4f} D_score:{d_fake_score:.4f}"
                print(log_info)
                with open(LOG_FILE, "a") as f: f.write(log_info + "\n")

        # 每 20 轮保存一次
        if (epoch + 1) % 20 == 0:
            torch.save(G.state_dict(), os.path.join(SAVE_DIR, f"G_baseline_epoch_{epoch + 1}.pth"))

    print("训练完成！请进行后续性能指标评测。")
