import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
import os
import time
from sklearn.cluster import KMeans
from matplotlib.colors import hsv_to_rgb

CURRENT_TIME = time.strftime(
    '%Y%m%d_%H_%M_%S', time.localtime(int(time.time())))
print(CURRENT_TIME)
    
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
    colors = image.getcolors(image.width * image.height)
    bins = color_filter(colors)

    rgb_points_list = []
    hsv_points_list = []
    for color, count in bins.items():
        rgb_color = list(color)
        hsv_color = rgb2hsv(np.array([[rgb_color]]))[0][0]
        hsv_color = [hsv_color[0] * 360, hsv_color[1] * 100, hsv_color[2] * 100]

        # 将整个HSV颜色作为一个列表添加
        for _ in range(count):
            hsv_points_list.append(hsv_color)
            rgb_points_list.append(rgb_color)
    
    return hsv_points_list, rgb_points_list

def extract_hsv_values(img_path):
    image = Image.open(img_path).convert("RGB")
    np_img = np.array(image)
    hsv_img = rgb2hsv(np_img)

    hue_values = hsv_img[:, :, 0].flatten() * 360  # Rescaling hue from 0-1 to 0-360
    saturation_values = hsv_img[:, :, 1].flatten() * 100  # Rescaling saturation to 0-100%
    value_values = hsv_img[:, :, 2].flatten() * 100  # Rescaling value to 0-100%

    return hue_values, saturation_values, value_values

def plot_histogram(values, title, bins, range, xlabel, save_dir, filename):
    plt.figure()
    plt.hist(values, bins=bins, range=range, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def analyze_and_plot_hsv(img_path, save_dir):
    hue, saturation, value = extract_hsv_values(img_path)

    # Plotting the histograms
    plot_histogram(hue, "Hue Distribution", 360, (0, 360), "Hue (Degrees)", save_dir, "hue_histogram.png")
    plot_histogram(saturation, "Saturation Distribution", 100, (0, 100), "Saturation (%)", save_dir, "saturation_histogram.png")
    plot_histogram(value, "Value Distribution", 100, (0, 100), "Value (%)", save_dir, "value_histogram.png")

def find_representative_color(hsv_points_list):
    kmeans = KMeans(n_clusters=1, random_state=0).fit(hsv_points_list)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color

def save_representative_color_image(hsv_color, save_path):
    # Ensure the HSV values are in the range [0-1, 0-1, 0-1]
    hsv_color_scaled = np.array([hsv_color[0] / 360, hsv_color[1] / 100, hsv_color[2] / 100])
    rgb_color = hsv_to_rgb([hsv_color_scaled])
    plt.figure(figsize=(1, 1), dpi=100)
    plt.imshow([[rgb_color[0]]])
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
    
#def batch_color_extraction(prompts, src_img_dir, save_dir):
def batch_color_extraction_and_representative_image(prompts, src_img_dir, save_dir):    
    if not isinstance(prompts, list):
        prompts = [prompts]
        
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
            hsv, rgb = color_extraction(img_path)
            hsv_points_list += hsv
            rgb_points_list += rgb

        # Save HSV and RGB points as .npy files
        np.save(os.path.join(palette_dir, "hsv_points_full_list.npy"), np.array(hsv_points_list))
        np.save(os.path.join(palette_dir, "rgb_points_full_list.npy"), np.array(rgb_points_list))

        # Find representative color after collecting all HSV points
        representative_color = find_representative_color(np.array(hsv_points_list))
        save_representative_color_image(representative_color, os.path.join(palette_dir, prompt + "_representative_color.png"))


        
# Example usage
prompts = ["apple","banana","buleberry","cherry","grape","peach","tangerine"],
OUTPUT_PARENT_PATH = rf'E:\Test\Testoutput{CURRENT_TIME}'
src_img_dir = rf'{OUTPUT_PARENT_PATH}/image'
save_dir = rf'{OUTPUT_PARENT_PATH}/color_stat'
