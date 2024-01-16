import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import hsv_to_rgb
import os
import time

CURRENT_TIME = time.strftime(
    '%Y%m%d_%H_%M_%S', time.localtime(int(time.time())))
print(CURRENT_TIME)

def color_filter(colors, min_count=2, max_black_threshold=50, min_brightness=50, max_white_threshold=240):
    bins = {}
    for count, pixel in colors:
        # 忽略黑色和非常暗的颜色
        if all(value < max_black_threshold for value in pixel) or sum(pixel) < min_brightness:
            continue
        # 忽略白色和非常亮的颜色
        if all(value > max_white_threshold for value in pixel):
            continue
        if count >= min_count:
            bins[pixel] = count
    return bins

def color_extraction_with_filter(img_path):
    image = Image.open(img_path).convert("RGB")
    colors = image.getcolors(image.width * image.height)
    filtered_colors = color_filter(colors)

    hsv_points_list = []
    rgb_points_list = []
    for color, count in filtered_colors.items():
        rgb_color = list(color)
        hsv_color = rgb2hsv(np.array([[rgb_color]]))[0][0]
        hsv_color = [hsv_color[0], hsv_color[1], hsv_color[2]]

        for _ in range(count):
            hsv_points_list.append(hsv_color)
            rgb_points_list.append(rgb_color)
    
    print("Sample filtered colors (RGB):", rgb_points_list[:10])
    print("Sample filtered colors (HSV):", hsv_points_list[:10])
    
    return hsv_points_list, rgb_points_list

def find_representative_color(rgb_points_list):
    # 使用 RGB 数据进行 KMeans 聚类
    kmeans = KMeans(n_clusters=1, random_state=0).fit(rgb_points_list)
    dominant_color = kmeans.cluster_centers_[0]
    
    print("Representative color (RGB):", dominant_color)
    
    return dominant_color

def save_representative_color_image(rgb_color, save_path):
    # 确保 RGB 值在 [0, 1] 范围内
    rgb_color_normalized = [x / 255 for x in rgb_color]
    plt.figure(figsize=(1, 1), dpi=100)
    plt.imshow([[rgb_color_normalized]])
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

# 在批处理函数中调用以上函数
# ...
def process_image_representative_color(img_path, save_path):
    # 提取颜色
    _, rgb = color_extraction_with_filter(img_path)
    # 找到代表性颜色
    representative_color = find_representative_color(np.array(rgb))
    # 保存代表性颜色为图像
    save_representative_color_image(representative_color, save_path)


def batch_representative_color(prompts, src_img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for prompt in prompts:
        prompt_dir = os.path.join(src_img_dir, prompt)  # 创建指向 prompt 子文件夹的路径
        if not os.path.exists(prompt_dir):  # 检查路径是否存在
            continue
        for file_name in os.listdir(prompt_dir):  # 遍历 prompt 子文件夹
            if not file_name.endswith(".png"):
                continue

            img_path = os.path.join(prompt_dir, file_name)  # 编辑这行，使用 prompt_dir
            # 构建保存代表性颜色图像的路径
            color_image_name = file_name.replace(".png", "_representative_color.png")
            save_path = os.path.join(save_dir, color_image_name)
            # 处理当前图片并保存代表性颜色
            process_image_representative_color(img_path, save_path)



#prompts = ["apple", "banana", "blueberry", "cherry", "grape", "peach", "tangerine"]
#OUTPUT_PARENT_PATH = rf'E:\Test\Testoutput{CURRENT_TIME}'
#src_img_dir = rf'{OUTPUT_PARENT_PATH}/image'
#save_dir = rf'{OUTPUT_PARENT_PATH}/color_stat'
# 示例使用
# hsv_points_list, _ = color_extraction("path_to_image.png")
# dominant_color = find_representative_color(hsv_points_list)
# save_representative_color_image(dominant_color, "path_to_save_representative_color.png")
