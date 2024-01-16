import sys
sys.path.append("/home/amax/Documents/semantic-color-code")
sys.path.append(r"E:\Test\Testcode\GroundSAM\Grounded-Segment-Anything")
from visualization import batch_plot_hue
from image_generation import batch_image_generation
from represent_color import batch_representative_color
from hsv_extraction import batch_hsv_extraction
from segment import batch_segment
import time
import random

CURRENT_TIME = time.strftime(
    '%Y%m%d_%H_%M_%S', time.localtime(int(time.time())))
print(CURRENT_TIME)
#CURRENT_TIME = "20221214-15:26:25"

# Notice: Change the path information to locate your outputs, defaultly set to current time.
# images, images with background removed, color statistics and results of visualization will be stored at
# OUTPUT_PATH/image, OUTPUT_PATH/image_rembg, OUTPUT_PATH/color_stat, OUTPUT_PATH/color_vis, respectively.

OUTPUT_PARENT_PATH = rf'E:\Test\Testoutput{CURRENT_TIME}'


def main():
    # arguments assigned
    prompts = ["horses in grassland"]
    # seeds = list(range(42,47))
    # scales = list(range(5,18,4))
    seeds = [random.randint(0, 2**32 - 1) for _ in range(5)]
    scales = 5
    n_samples = 1
    
    batch_image_generation(prompts, 
                           save_dir=rf'{OUTPUT_PARENT_PATH}/image', 
                           n_samples=n_samples, scales=scales,seeds=seeds)

    batch_segment(prompts,
                           src_dir=rf'{OUTPUT_PARENT_PATH}/image',
                           save_dir=rf'{OUTPUT_PARENT_PATH}/segment')
    
    batch_representative_color(prompts,
                           src_img_dir=rf'{OUTPUT_PARENT_PATH}/segment',
                           save_dir=rf'{OUTPUT_PARENT_PATH}/color_stat')
    
    batch_hsv_extraction(prompts,
                        src_img_dir=rf'{OUTPUT_PARENT_PATH}/segment',
                        save_dir=rf'{OUTPUT_PARENT_PATH}/color_stat')
    
    batch_plot_hue(prompts,
                   src_dir=rf'{OUTPUT_PARENT_PATH}/color_stat',
                   save_dir=rf'{OUTPUT_PARENT_PATH}/color_vis')



if __name__ == "__main__":
    main()
