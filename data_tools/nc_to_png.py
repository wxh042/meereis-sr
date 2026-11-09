import netCDF4 as nc
import numpy as np
from PIL import Image
import os

# --- 路径配置 ---
input_dir = os.getenv("MEEREIS_NC_DIR", "datasets/raw_nc")
output_dir = os.getenv("MEEREIS_HR_DIR", "datasets/train_HR")
var_name = "cdr_seaice_conc"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def process():
    files = [f for f in os.listdir(input_dir) if f.endswith('.nc')]
    print(f"找到 {len(files)} 个文件，开始处理...")

    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        ds = nc.Dataset(file_path)

        # 1. 提取数据 (取出第一层 [0, y, x])
        # 注意：这里要查看数据的维度，通常是 (time, y, x)
        raw_data = ds.variables[var_name][0]

        # 2. 处理掩码 (关键步骤！)
        # 根据 NSIDC 标准：
        # 0-1.0 是正常海冰数据
        # > 2.5 (或者 > 250) 的是陆地、极点孔等
        # 我们把陆地设为 0 (和海洋一样)，或者设为 1 (和厚冰一样)
        # 本科论文建议：把陆地设为 0，只关注有冰的区域
        data = raw_data.filled(0)  # 处理 masked array

        # 逻辑：保留 0-1 之间的数据，其余设为 0
        data = np.where((data >= 0) & (data <= 1.0), data, 0)

        # 3. 归一化到 0-255
        # 0.0 (水) -> 0
        # 1.0 (全冰) -> 255
        img_array = (data * 255).astype(np.uint8)

        # 4. 保存图片
        # 原始尺寸应该是 448x304
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, file_name.replace('.nc', '.png')))

        ds.close()
        print(f"已生成图像: {file_name.replace('.nc', '.png')}")


if __name__ == "__main__":
    process()