from __future__ import annotations

import csv
import os
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover - optional dependency fallback
    skimage_ssim = None


ROOT = Path(__file__).resolve().parent
TEST_ROOT = Path(os.getenv("MEEREIS_TEST_ROOT", ROOT.parent / "datasets" / "test_202301"))
LR_DIR = TEST_ROOT / "test_LR"
HR_DIR = TEST_ROOT / "test_HR"
WEIGHT_PATH = ROOT.parent / "platform" / "G_v4_epoch_100.pth"

OUTPUT_DIR = ROOT / "evaluation_outputs_10km"
SR_DIR = OUTPUT_DIR / "SR_Result"
BICUBIC_DIR = OUTPUT_DIR / "Bicubic_Result"
METRICS_CSV = ROOT / "evaluation_metrics_10km.csv"
MATRIX_PATH = ROOT / "evaluation_matrix.png"


class CBAM(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ca(x) * x
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        spatial = self.sa(torch.cat([avg_out, max_out], dim=1))
        return spatial * out


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            CBAM(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels: int, up_scale: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (up_scale**2), 3, 1, 1),
            nn.PixelShuffle(up_scale),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, scale_factor: int):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 64, 9, 1, 4), nn.PReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(6)])
        self.block7 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.upsample = nn.Sequential(
            *[UpsampleBLock(64, 2) for _ in range(int(math.log2(scale_factor)))]
        )
        self.block8 = nn.Conv2d(64, 1, 9, 1, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.block1(x)
        res = self.res_blocks(b1)
        b7 = self.block7(res)
        out = self.block8(self.upsample(b1 + b7))
        return (torch.tanh(out) + 1) / 2


def load_generator(weight_path: Path, device: torch.device) -> Generator:
    model = Generator(scale_factor=4).to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and "G" in checkpoint:
        model.load_state_dict(checkpoint["G"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def image_to_float_array(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def lr_to_tensor(path: Path) -> torch.Tensor:
    arr = image_to_float_array(path) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    if skimage_ssim is not None:
        return float(skimage_ssim(target, pred, data_range=255))

    pred_f = pred.astype(np.float64)
    target_f = target.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_x = pred_f.mean()
    mu_y = target_f.mean()
    sigma_x = pred_f.var()
    sigma_y = target_f.var()
    sigma_xy = ((pred_f - mu_x) * (target_f - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / denominator)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "rmse": rmse(pred, target),
    }


def representative_samples(paired_files: list[Path]) -> list[tuple[str, str]]:
    records = []
    for lr_path in paired_files:
        hr_arr = image_to_float_array(HR_DIR / lr_path.name)
        gy, gx = np.gradient(hr_arr)
        grad = np.mean(np.sqrt(gx * gx + gy * gy))
        records.append(
            {
                "name": lr_path.name,
                "mean": float(hr_arr.mean()),
                "std": float(hr_arr.std()),
                "grad": float(grad),
            }
        )

    high = max(records, key=lambda r: (r["mean"], r["std"]))
    edge_candidates = [r for r in records if 60 <= r["mean"] <= 190]
    edge = max(edge_candidates or records, key=lambda r: (r["grad"] + 0.25 * r["std"]))
    broken_candidates = [
        r for r in records if 25 <= r["mean"] <= 225 and r["name"] not in {high["name"], edge["name"]}
    ]
    broken = max(broken_candidates or records, key=lambda r: (r["std"], r["grad"]))

    return [
        ("高密集度海冰区", high["name"]),
        ("冰缘过渡区", edge["name"]),
        ("破碎流冰区", broken["name"]),
    ]


def plot_evaluation_matrix(
    selected: list[tuple[str, str]],
    metrics_by_name: dict[str, dict[str, float]],
) -> None:
    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 8.6), dpi=300)
    col_titles = ["LR输入", "Bicubic插值", "SRGAN+CBAM重建", "HR真值"]
    for row, (label, name) in enumerate(selected):
        lr_img = Image.open(LR_DIR / name).convert("L")
        hr_img = Image.open(HR_DIR / name).convert("L")
        bicubic_img = Image.open(BICUBIC_DIR / name).convert("L")
        sr_img = Image.open(SR_DIR / name).convert("L")
        m = metrics_by_name[name]

        images = [lr_img, bicubic_img, sr_img, hr_img]
        metric_labels = [
            "",
            f"PSNR: {m['bicubic_psnr']:.2f} dB | SSIM: {m['bicubic_ssim']:.3f}",
            f"PSNR: {m['sr_psnr']:.2f} dB | SSIM: {m['sr_ssim']:.3f}",
            "",
        ]
        for col, (ax, img) in enumerate(zip(axes[row], images)):
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, fontweight="bold", pad=8)
            if metric_labels[col]:
                ax.set_xlabel(metric_labels[col], fontsize=8.2, labelpad=5)

        axes[row, 0].set_ylabel(label, fontsize=10.5, fontweight="bold", labelpad=12)
        axes[row, 0].set_xlabel(name, fontsize=6.8, labelpad=5)

    fig.suptitle("重建效果对比", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.15, w_pad=0.6)
    fig.savefig(MATRIX_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not LR_DIR.exists() or not HR_DIR.exists():
        raise FileNotFoundError(f"Missing test directories: {LR_DIR} or {HR_DIR}")
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"Missing model weight: {WEIGHT_PATH}")

    SR_DIR.mkdir(parents=True, exist_ok=True)
    BICUBIC_DIR.mkdir(parents=True, exist_ok=True)

    lr_files = sorted(LR_DIR.glob("*.png"))
    hr_names = {p.name for p in HR_DIR.glob("*.png")}
    paired_files = [p for p in lr_files if p.name in hr_names]

    print(f"LR files: {len(lr_files)}")
    print(f"HR files: {len(hr_names)}")
    print(f"Paired files: {len(paired_files)}")
    if not paired_files:
        raise RuntimeError("No paired LR/HR PNG files found.")

    selected = representative_samples(paired_files)
    print("Selected representative samples:")
    for label, name in selected:
        print(f"  - {label}: {name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_generator(WEIGHT_PATH, device)

    rows = []
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(paired_files), batch_size):
            batch_files = paired_files[start : start + batch_size]
            lr_batch = torch.stack([lr_to_tensor(p) for p in batch_files], dim=0).to(device)
            sr_batch = model(lr_batch).squeeze(1).cpu().numpy()

            for idx, lr_path in enumerate(batch_files):
                name = lr_path.name
                hr_arr = image_to_float_array(HR_DIR / name)
                lr_img = Image.open(lr_path).convert("L")

                bicubic = np.array(lr_img.resize(hr_arr.shape[::-1], Image.BICUBIC), dtype=np.float32)
                sr = to_uint8(sr_batch[idx] * 255.0)
                bicubic_u8 = to_uint8(bicubic)
                hr_u8 = to_uint8(hr_arr)

                Image.fromarray(sr, mode="L").save(SR_DIR / name)
                Image.fromarray(bicubic_u8, mode="L").save(BICUBIC_DIR / name)

                bicubic_metrics = compute_metrics(bicubic_u8, hr_u8)
                sr_metrics = compute_metrics(sr, hr_u8)
                rows.append(
                    {
                        "filename": name,
                        "bicubic_psnr": bicubic_metrics["psnr"],
                        "bicubic_ssim": bicubic_metrics["ssim"],
                        "bicubic_rmse": bicubic_metrics["rmse"],
                        "sr_psnr": sr_metrics["psnr"],
                        "sr_ssim": sr_metrics["ssim"],
                        "sr_rmse": sr_metrics["rmse"],
                    }
                )

            print(f"Processed {min(start + batch_size, len(paired_files))}/{len(paired_files)}")

    with METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics_by_name = {row["filename"]: row for row in rows}
    plot_evaluation_matrix(selected, metrics_by_name)

    summary = {
        "Bicubic": {
            "PSNR": float(np.mean([r["bicubic_psnr"] for r in rows])),
            "SSIM": float(np.mean([r["bicubic_ssim"] for r in rows])),
            "RMSE": float(np.mean([r["bicubic_rmse"] for r in rows])),
        },
        "SRGAN+CBAM": {
            "PSNR": float(np.mean([r["sr_psnr"] for r in rows])),
            "SSIM": float(np.mean([r["sr_ssim"] for r in rows])),
            "RMSE": float(np.mean([r["sr_rmse"] for r in rows])),
        },
    }

    print("\nAverage metrics over all paired samples")
    print("Method,PSNR(dB),SSIM,RMSE")
    for method, values in summary.items():
        print(f"{method},{values['PSNR']:.4f},{values['SSIM']:.4f},{values['RMSE']:.4f}")
    print(f"Metrics CSV: {METRICS_CSV}")
    print(f"Visual matrix: {MATRIX_PATH}")
    print(f"SR outputs: {SR_DIR}")
    print(f"Bicubic outputs: {BICUBIC_DIR}")


if __name__ == "__main__":
    main()
