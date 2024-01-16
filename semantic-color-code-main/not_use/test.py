import numpy as np

# 定义文件路径
file_path = r'E:\Test\Testoutput20231212_21_36_50\color_stat\Rainy sky\rgb_points_full_list.npy'

# 载入文件
rgb_points = np.load(file_path)

# 输出基本信息
print(f"Shape of the data: {rgb_points.shape}")
print(f"First 10 data points:\n{rgb_points[:10]}")

# 检查数据的范围和分布
print(f"Data Range:\nMin: {rgb_points.min(axis=0)}\nMax: {rgb_points.max(axis=0)}")
