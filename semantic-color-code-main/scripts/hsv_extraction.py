import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
import os
import time

CURRENT_TIME = time.strftime(
    '%Y%m%d_%H_%M_%S', time.localtime(int(time.time())))
print(CURRENT_TIME)

def extract_hsv_values(img_path):
    image = Image.open(img_path).convert("RGB")
    np_img = np.array(image)
    hsv_img = rgb2hsv(np_img)

    hue_values = hsv_img[:, :, 0].flatten() * 360  # Rescaling hue from 0-1 to 0-360
    saturation_values = hsv_img[:, :, 1].flatten() * 100  # Rescaling saturation to 0-100%
    value_values = hsv_img[:, :, 2].flatten() * 100  # Rescaling value to 0-100%

    return hue_values, saturation_values, value_values

def color_filter(colors, min_count=5, max_black_threshold=30, min_brightness=15):
    """
    Filters out colors based on specific conditions.

    :param colors: A list of tuples, where each tuple contains the count and the RGB value.
    :param min_count: Minimum count for a color to be considered.
    :param max_black_threshold: Maximum value for each RGB component to be considered black.
    :param min_brightness: Minimum sum of RGB values to consider a color not too dark.
    :return: Dictionary of filtered colors with their counts.
    """
    bins = {}

    for count, pixel in colors:
        # Filter out very dark/black colors
        if all(value < max_black_threshold for value in pixel):
            continue

        # Filter out very dim colors
        if sum(pixel) < min_brightness:
            continue

        # Only consider colors with a count higher than min_count
        if count >= min_count:
            bins[pixel] = count

    return bins

def color_extraction(img_path):
    image = Image.open(img_path).convert("RGB")
    np_img = np.array(image)
    # 获取唯一颜色和对应的计数
    unique_colors, counts = np.unique(np_img.reshape(-1, 3), axis=0, return_counts=True)
    # 将颜色和计数整合成 (计数, 颜色) 对的列表
    colors = list(zip(counts, map(tuple, unique_colors)))
    # 使用 color_filter 函数应用颜色过滤规则
    bins = color_filter(colors)

    rgb_points_list = []
    hsv_points_list = []
    for count, color in bins.items():
        rgb_color = list(color)
        hsv_color = rgb2hsv(np.array([[rgb_color]]))[0][0]
        hsv_color = [hsv_color[0] * 360, hsv_color[1] * 100, hsv_color[2] * 100]

        # 将整个 HSV 颜色作为一个列表添加
        for _ in range(count):
            hsv_points_list.append(hsv_color)
            rgb_points_list.append(rgb_color)
    
    return hsv_points_list, rgb_points_list
def plot_histogram(values, title, bins, range, xlabel, save_dir, filename):
    plt.figure()
    plt.hist(values, bins=bins, range=range, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def analyze_and_plot_combined_hsv(img_path, save_dir, filename):
    hue, saturation, value = extract_hsv_values(img_path)

 # Calculate the maximum frequency for normalization
    max_frequency_hue = max(np.histogram(hue, bins=360, range=(0, 360))[0])
    max_frequency_saturation = max(np.histogram(saturation, bins=100, range=(0, 100))[0])
    max_frequency_value = max(np.histogram(value, bins=100, range=(0, 100))[0])

    max_frequency = max(max_frequency_hue, max_frequency_saturation, max_frequency_value)
    
    plt.figure(figsize=(18, 6))

    # Plotting the hue histogram
    plt.subplot(1, 3, 1)
    plt.hist(hue, bins=360, range=(0, 360), color='red', alpha=0.7)
    plt.title("Hue Distribution")
    plt.xlabel("Hue (Degrees)")
    plt.ylabel("Frequency")

    # Plotting the saturation histogram
    plt.subplot(1, 3, 2)
    plt.hist(saturation, bins=100, range=(0, 100), color='green', alpha=0.7)
    plt.title("Saturation Distribution")
    plt.xlabel("Saturation (%)")

    # Plotting the value histogram
    plt.subplot(1, 3, 3)
    plt.hist(value, bins=100, range=(0, 100), color='blue', alpha=0.7)
    plt.title("Value Distribution")
    plt.xlabel("Value (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def batch_hsv_extraction(prompts, src_img_dir, save_dir):
    for prompt in prompts:
        image_dir = os.path.join(src_img_dir, prompt)
        palette_dir = os.path.join(save_dir, prompt)
        os.makedirs(palette_dir, exist_ok=True)

        hsv_points_list = []
        rgb_points_list = []

        for file_name in os.listdir(image_dir):
            if not file_name.endswith(".png"):
                continue

            img_path = os.path.join(image_dir, file_name)
            combined_filename = os.path.splitext(file_name)[0] + "_combined_histogram.png"
            combined_save_path = os.path.join(palette_dir, combined_filename)  # 更新保存路径为 palette_dir
            
            analyze_and_plot_combined_hsv(img_path, file_name, combined_save_path)  # 更新函数参数，传递正确的保存路径

            # 提取并累积HSV和RGB值
            hsv, rgb = color_extraction(img_path)
            hsv_points_list += hsv
            rgb_points_list += rgb

        np.save(os.path.join(palette_dir, "hsv_points_full_list.npy"), np.array(hsv_points_list))
        np.save(os.path.join(palette_dir, "rgb_points_full_list.npy"), np.array(rgb_points_list))




#prompts = ["apple","banana","buleberry","cherry","grape","peach","tangerine"],
#OUTPUT_PARENT_PATH = rf'E:\Test\Testoutput{CURRENT_TIME}'
#src_img_dir = rf'{OUTPUT_PARENT_PATH}/image'
#save_dir = rf'{OUTPUT_PARENT_PATH}/color_stat'
# 示例使用
# analyze_and_plot_hsv("path_to_image.png", "path_to_save_histograms")
