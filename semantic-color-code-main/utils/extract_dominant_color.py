from PIL import Image
import sys
sys.path.append('E:\Test\Testcode\semantic-color-code-main\utils')
from palette import *
from util import *
import os
import numpy as np

IMAGE_PARENT_DIR = r"E:\Test\Testoutput\image_12_12"
PALETTE_PARENT_DIR = r"E:\Test\Testoutput\palette_12_12"

#TODO Use DBSCAN to cluster the dominant colors extracted from generated images.
def dominant_color_extraction(image):
    colors = build_palette(image, random_init=False, black=True)
    # print(len(colors))
    # h, s, v = colors[0]['hsv']
    # r, g, b = colors[0]['rgb']
    hsv_list = [colors[i]['hsv'] for i in range(len(colors)) if colors[i]['size'] > 20]
    rgb_list = [colors[i]['rgb'] for i in range(len(colors)) if colors[i]['size'] > 20]
    # hsv_list = colors[:]['hsv']
    # rgb_list = colors[:]['rgb']
    # print(hsv_list)
    # raise OSError
    # return [h, s, v], [r, g, b]
    return hsv_list, rgb_list


def extract_dominant_color(prompts, img_path=IMAGE_PARENT_DIR):
    # generate more photos
    if not isinstance(prompts, list):
        prompts = [prompts]
    for prompt in prompts:
        image_dir = f"{IMAGE_PARENT_DIR}/{prompt}/samples"
        palette_dir = f"{PALETTE_PARENT_DIR}/{prompt}"
        hsv_points_list = []
        rgb_points_list = []
        for file_name in os.listdir(image_dir):
            image_path = f"{image_dir}/{file_name}"
            # print(image_path)
            if image_path.endswith(".png"):
                image = Image.open(image_path)
                hsv, rgb = dominant_color_extraction(image)
                if isinstance(hsv,list):
                    hsv_points_list += hsv
                elif isinstance(hsv, tuple):
                    hsv_points_list.append(hsv)
                else:
                    raise RuntimeError('Unexcepted input of hsv')
                
                if isinstance(rgb,list):
                    rgb_points_list += rgb
                elif isinstance(hsv, tuple):
                    rgb_points_list.append(rgb)
                else:
                    raise RuntimeError('Unexcepted input of rgb')
                                
        hsv_points_list = np.array(hsv_points_list)
        rgb_points_list = np.array(rgb_points_list)
        print(hsv_points_list)
        # TODO DBSCAN?
        os.makedirs(f"{palette_dir}", exist_ok=True)
        # np.save(f"{palette_dir}/hsv_points_list.npy", hsv_points_list)
        # np.save(f"{palette_dir}/rgb_points_list.npy", rgb_points_list)
        np.save(f"{palette_dir}/hsv_points_full_list.npy", hsv_points_list)
        np.save(f"{palette_dir}/rgb_points_full_list.npy", rgb_points_list)


if __name__ == '__main__':    
    prompts = ["banana, fruit"]
    extract_dominant_color(prompts)

