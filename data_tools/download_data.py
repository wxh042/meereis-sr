import os
import requests
import netCDF4 as nc
import numpy as np
from PIL import Image
import time

# ========================================================
# 1. 路径配置 (依然复用旧目录)
# ========================================================
BASE_DIR = "osi_saf_massive_dataset"
NC_DIR = os.path.join(BASE_DIR, "raw_nc")
PATCH_HR_DIR = os.path.join(BASE_DIR, "train_HR")
PATCH_LR_DIR = os.path.join(BASE_DIR, "train_LR")

for d in [NC_DIR, PATCH_HR_DIR, PATCH_LR_DIR]:
    os.makedirs(d, exist_ok=True)

# 2. 任务设计：6个月份，每个月30天，覆盖2022年全年的科学样本
TASKS = [



    ("2022", "07", 15),  # 7月：盛夏破碎 (核心挑战点)
    ("2022", "09", 15),  # 9月：海冰最低值期 (核心挑战点)
    ("2022", "10", 15)  # 11月：快速封冻期
]

VAR_NAME = "ice_conc"
PATCH_SIZE = 128
SCALE = 4
STRIDE = 48  # 【最优步长】确保充足样本量且不过拟合
LR_SIZE = PATCH_SIZE // SCALE


def download_osi_file(year, month, day):
    date_str = f"{year}{month}{day}"
    filename = f"ice_conc_nh_polstere-100_multi_{date_str}1200.nc"
    url = f"https://thredds.met.no/thredds/fileServer/osisaf/met.no/ice/conc/{year}/{month}/{filename}"
    local_path = os.path.join(NC_DIR, filename)

    if os.path.exists(local_path):
        return local_path

    print(f"-> 正在下载 {date_str}...")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(r.content)
            return local_path
        return None
    except:
        return None


def slice_nc_file(nc_path):
    try:
        ds = nc.Dataset(nc_path)
        raw_data = ds.variables[VAR_NAME][0]
        data = raw_data.filled(0).astype(np.float32)
        data = np.where((data >= 0) & (data <= 100), data, 0)

        hr_full = (data * 2.55).astype(np.uint8)
        hr_pil = Image.fromarray(hr_full, mode='L')
        w, h = hr_pil.size
        lr_full_arr = np.array(hr_pil.resize((w // SCALE, h // SCALE), Image.BICUBIC))

        count = 0
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                patch_hr = hr_full[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

                # 过滤：平均密集度灰度大于 10 的保留。
                # 这个阈值能保留绝大部分有意义的冰缘细节
                if np.mean(patch_hr) < 10:
                    continue

                y_lr, x_lr = y // SCALE, x // SCALE
                patch_lr = lr_full_arr[y_lr:y_lr + LR_SIZE, x_lr:x_lr + LR_SIZE]

                fn_base = os.path.basename(nc_path).replace(".nc", "")
                patch_name = f"{fn_base}_y{y}_x{x}.png"

                Image.fromarray(patch_hr).save(os.path.join(PATCH_HR_DIR, patch_name))
                Image.fromarray(patch_lr).save(os.path.join(PATCH_LR_DIR, patch_name))
                count += 1
        ds.close()
        return count
    except:
        return 0


if __name__ == "__main__":
    print("=== 启动学术级全季节海冰数据集构建 ===")
    total_count = 0
    start_time = time.time()

    for year, month, days in TASKS:
        print(f"\n任务阶段: {year}年{month}月")
        for d in range(1, days + 1):
            day_str = str(d).zfill(2)
            path = download_osi_file(year, month, day_str)
            if path:
                num = slice_nc_file(path)
                total_count += num
                print(f"   {day_str}日: 新增 {num} | 总数: {total_count}")

    print(f"\n任务完成！总计样本量: {total_count} 对")
    print(f"总耗时: {(time.time() - start_time) / 60:.1f} 分钟")