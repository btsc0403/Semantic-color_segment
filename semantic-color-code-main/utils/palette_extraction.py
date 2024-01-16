import sys
from unittest import skip
from PIL import Image
from utils.palette import *
from utils.util import *
import json
import os

img_dir = r'E:\Test\Testoutput\image_12_12\grape, fruit\samples'
palette_dir = r'E:\Test\Testoutput\palette_12_12\test\grape'

assert os.path.exists(img_dir)
if not os.path.exists(palette_dir):
    os.makedirs(palette_dir)
    
file_indices = list(range(0, 10))

def palette_generation(image, txt_path):
    colors_list = []
    try:
        os.remove(txt_path)
    except OSError:
        pass

    for random_init, black in itertools.product([True, False], repeat=2):
        print('random_init: {}, black: {}'.format(random_init, black))
        colors = build_palette(image, random_init=random_init, black=black)
        
        #TODO when will build_palette return None?
        if colors == None:
            skip
        colors_list.append(draw_palette(colors))
        
        with open(txt_path, 'a') as fout:
            json.dump(colors, fout)
            fout.write("\n")

    return v_merge(colors_list)

if __name__ == '__main__':
    for file_index in file_indices:
        # os preprocess
        file_name = "{:05d}".format(file_index) +'.png'
        img_path = os.path.join(img_dir, file_name)
        palette_path = os.path.join(palette_dir, "palette_"+"{:05d}".format(file_index) +'.png')
        
        image = Image.open(img_path)
        print(img_path, image.format, image.size, image.mode)
        # e.g. .../00000.png PNG (512, 512) RGB
        palette_img = palette_generation(rgb2lab(image), palette_path.replace("png", "txt"))
        assert palette_img is not None
        palette_img.save(palette_path)