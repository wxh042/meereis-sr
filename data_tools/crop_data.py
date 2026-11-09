import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


def inspect_and_plot_he5(file_path):
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    # 1. 打开文件
    with h5py.File(file_path, 'r') as f:
        print("--- 文件结构探测 ---")

        # 递归打印所有数据集路径，帮你找到正确的变量名
        dataset_paths = []

        def find_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                dataset_paths.append(name)

        f.visititems(find_datasets)

        # 打印前10个找到的路径供参考
        for path in dataset_paths[:10]:
            print(f"发现数据集: {path}")

        # 2. 匹配海冰密集度变量 (通常包含 'ICECON' 或 'Concentration')
        # AMSR2 典型路径: 'HDFEOS/GRIDS/NpPolarGrid12km/Data Fields/SI_12km_NH_ICECON_DAY'
        target_var = None
        for path in dataset_paths:
            if 'ICECON' in path or 'SeaIce' in path or 'Concentration' in path:
                target_var = path
                break

        if not target_var:
            print("未找到明显的 SIC 变量，请手动检查打印的路径。")
            return

        print(f"\n正在读取变量: {target_var}")
        data = f[target_var][:]

        # 3. 数据清洗
        # 通常 HE5 里的 SIC 范围是 0-1000 或 0-100，且有填充值 (如 -9999)
        data = data.astype(float)
        fill_value = -9999  # 或者查看 f[target_var].attrs['_FillValue']

        # 将无效值设为 NaN，方便绘图展示为透明或黑色
        data[data < 0] = np.nan
        data[data > 1000] = np.nan  # 针对 0-1000 范围的缩放

        # 某些产品单位是 0.1%，需要除以 10
        if np.nanmax(data) > 100:
            data = data / 10.0

        # 4. 可视化
        plt.figure(figsize=(10, 8))
        # 使用 'icefire' 或 'Blues_r' 比较符合海冰视觉
        im = plt.imshow(data, cmap='viridis', origin='upper')
        plt.colorbar(im, label='Sea Ice Concentration (%)')

        plt.title(f"AMSR2 Sea Ice Concentration (12.5km)\n{os.path.basename(file_path)}")
        plt.axis('off')  # 隐藏像素坐标

        # 5. 保存并展示
        output_name = "amsr2_check.png"
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"\n结果已保存至: {output_name}")
        plt.show()


# --- 使用你的路径 ---
file_path = os.getenv("MEEREIS_HE5_FILE", "datasets/sample.he5")
inspect_and_plot_he5(file_path)