import os
import requests
import netCDF4 as nc
import numpy as np
from PIL import Image
import time

# ========================================================
# 配置
# ========================================================
BASE_DIR = "osi_saf_test_202301"          # 测试集专用文件夹
NC_DIR = os.path.join(BASE_DIR, "raw_nc")
TEST_HR_DIR = os.path.join(BASE_DIR, "test_HR")
TEST_LR_DIR = os.path.join(BASE_DIR, "test_LR")

for d in [NC_DIR, TEST_HR_DIR, TEST_LR_DIR]:
    os.makedirs(d, exist_ok=True)

# 任务：2023年1月全部31天
YEAR = "2023"
MONTH = "01"
DAYS = 31

VAR_NAME = "ice_conc"
PATCH_SIZE = 128          # 高分辨率块大小
SCALE = 4                 # 超分倍率
LR_SIZE = PATCH_SIZE // SCALE   # 32
STRIDE = PATCH_SIZE       # 测试集无重叠，完整覆盖

# 过滤阈值：平均密集度（灰度值）小于 10 的块丢弃，约为 3.9%
MIN_MEAN_GRAY = 10


def download_osi_file(year, month, day):
    """下载指定日期的 NC 文件，返回本地路径，失败返回 None"""
    date_str = f"{year}{month}{day}"
    filename = f"ice_conc_nh_polstere-100_multi_{date_str}1200.nc"
    url = f"https://thredds.met.no/thredds/fileServer/osisaf/met.no/ice/conc/{year}/{month}/{filename}"
    local_path = os.path.join(NC_DIR, filename)

    if os.path.exists(local_path):
        return local_path

    print(f"-> 正在下载 {date_str}...", end=" ")
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(r.content)
            print("成功")
            return local_path
        else:
            print(f"失败（状态码 {r.status_code}）")
            return None
    except Exception as e:
        print(f"失败（{e}）")
        return None


def slice_nc_file(nc_path):
    """从 NC 文件生成无重叠的 HR/LR 测试块，返回成功生成的块数"""
    try:
        ds = nc.Dataset(nc_path)
        raw_data = ds.variables[VAR_NAME][0]
        # 填充值与有效范围处理
        data = raw_data.filled(0).astype(np.float32)
        # 密集度应在 0-100 之间，其它值视为无效（置 0）
        data = np.where((data >= 0) & (data <= 100), data, 0)

        # 转换到 0-255 灰度空间（与训练集一致）
        hr_full = (data * 2.55).astype(np.uint8)
        hr_pil = Image.fromarray(hr_full, mode='L')
        w, h = hr_pil.size

        # 下采样得到整张低分辨率图（用于裁剪 LR 块）
        lr_full_pil = hr_pil.resize((w // SCALE, h // SCALE), Image.BICUBIC)
        lr_full_arr = np.array(lr_full_pil)

        count = 0
        # 无重叠滑窗
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                patch_hr = hr_full[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

                # 丢弃几乎纯海的块
                if np.mean(patch_hr) < MIN_MEAN_GRAY:
                    continue

                # 对应的 LR 块
                y_lr, x_lr = y // SCALE, x // SCALE
                patch_lr = lr_full_arr[y_lr:y_lr + LR_SIZE, x_lr:x_lr + LR_SIZE]

                # 保存文件，命名包含日期与坐标信息
                fn_base = os.path.basename(nc_path).replace(".nc", "")
                patch_name = f"{fn_base}_y{y}_x{x}.png"

                Image.fromarray(patch_hr).save(os.path.join(TEST_HR_DIR, patch_name))
                Image.fromarray(patch_lr).save(os.path.join(TEST_LR_DIR, patch_name))
                count += 1

        ds.close()
        return count
    except Exception as e:
        print(f"    切片时出错: {e}")
        return 0


if __name__ == "__main__":
    print("=== 构建 2023 年 1 月测试集 ===")
    total_count = 0
    start_time = time.time()

    for day in range(1, DAYS + 1):
        day_str = str(day).zfill(2)
        nc_path = download_osi_file(YEAR, MONTH, day_str)
        if nc_path:
            num = slice_nc_file(nc_path)
            total_count += num
            print(f"   {day_str} 日: 新增 {num} 块 | 累计: {total_count}")
        else:
            print(f"   {day_str} 日: 下载失败，跳过")

    print(f"\n测试集构建完成！总计样本: {total_count} 对")
    print(f"总耗时: {(time.time() - start_time) / 60:.1f} 分钟")