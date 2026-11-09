import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
import math

# --- 自动路径 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['TORCH_HOME'] = os.path.join(BASE_DIR, "pretrained")


# ============================================================
# 模型定义：全部改为单通道 (Input: 1, Output: 1)
# ============================================================

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out)) * x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1);
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1);
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels)

    def forward(self, x):
        res = self.conv1(x);
        res = self.bn1(res);
        res = self.prelu(res)
        res = self.conv2(res);
        res = self.bn2(res);
        res = self.cbam(res)
        return x + res


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (up_scale ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale);
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 64, 9, 1, 4), nn.PReLU())  # 输入改为 1
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(6)])
        self.block7 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.upsample = nn.Sequential(*[UpsampleBLock(64, 2) for _ in range(int(math.log2(scale_factor)))])
        self.block8 = nn.Conv2d(64, 1, 9, 1, 4)  # 输出改为 1

    def forward(self, x):
        b1 = self.block1(x);
        res = self.res_blocks(b1);
        b7 = self.block7(res)
        out = self.block8(self.upsample(b1 + b7))
        return (torch.tanh(out) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.LeakyReLU(0.2),  # 输入改为 1
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(128, 256, 1), nn.LeakyReLU(0.2), nn.Conv2d(256, 1, 1)
        )

    def forward(self, x):
        return self.net(x).view(x.size(0))


# ============================================================
# 数据与损失
# ============================================================

class SRDataset(Dataset):
    def __init__(self, hr_dir, scale_factor=4, crop_size=128):
        self.hr_dir = hr_dir;
        self.scale_factor = scale_factor;
        self.crop_size = crop_size
        self.hr_images = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.lower().endswith('.png')]

    def __len__(self): return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image = Image.open(self.hr_images[idx]).convert('L')  # 强制灰度读取
        hr_tensor = transforms.ToTensor()(hr_image)
        lr_size = self.crop_size // self.scale_factor
        lr_pil = transforms.ToPILImage()(hr_tensor).resize((lr_size, lr_size), Image.BICUBIC)
        return transforms.ToTensor()(lr_pil), hr_tensor


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:35].eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg

    def forward(self, gen, target):
        # 核心修正：将单通道复制为三通道，以适配 VGG
        gen_3ch = torch.cat([gen, gen, gen], dim=1)
        target_3ch = torch.cat([target, target, target], dim=1)
        return torch.mean((self.vgg(gen_3ch) - self.vgg(target_3ch)) ** 2)


# ============================================================
# 训练循环 (优化了 G/D 平衡)
# ============================================================

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hr_dir = os.path.join(BASE_DIR, "train_patches")
    save_dir = os.path.join(BASE_DIR, "check_points_v2")  # 存到新文件夹
    os.makedirs(save_dir, exist_ok=True)

    G = Generator(4).to(device);
    D = Discriminator().to(device)
    vgg_l = VGGPerceptualLoss().to(device);
    mse_l = nn.MSELoss();
    bce_l = nn.BCEWithLogitsLoss()

    # 调低学习率，增加稳定性
    opt_G = optim.Adam(G.parameters(), lr=1e-4)
    opt_D = optim.Adam(D.parameters(), lr=5e-5)  # 让 D 慢一点

    loader = DataLoader(SRDataset(hr_dir), batch_size=64, shuffle=True, num_workers=4)

    for epoch in range(100):  # 练 100 轮
        for i, (lr_imgs, hr_imgs) in enumerate(loader):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # 训练 G
            opt_G.zero_grad()
            fake_hr = G(lr_imgs)
            # 增加 MSE 权重，减少对抗干扰，防止结构变形
            loss_g = 1.0 * mse_l(fake_hr, hr_imgs) + 0.006 * vgg_l(fake_hr, hr_imgs) + 1e-3 * bce_l(D(fake_hr),
                                                                                                    torch.ones_like(
                                                                                                        D(fake_hr)))
            loss_g.backward();
            opt_G.step()

            # 训练 D (增加判断条件，防止 D 太强)
            d_fake_score = D(fake_hr.detach()).mean().item()
            if d_fake_score > 0.1:  # 只有当 G 能稍微骗过 D 时才更新 D
                opt_D.zero_grad()
                loss_d = (bce_l(D(hr_imgs), torch.ones_like(D(hr_imgs))) + bce_l(D(fake_hr.detach()), torch.zeros_like(
                    D(fake_hr.detach())))) / 2
                loss_d.backward();
                opt_D.step()

            if i % 20 == 0:
                print(f"Epoch [{epoch + 1}] Batch {i} G:{loss_g.item():.4f} D_score:{d_fake_score:.4f}")

        if (epoch + 1) % 20 == 0:
            torch.save(G.state_dict(), f"{save_dir}/G_v2_epoch_{epoch + 1}.pth")
