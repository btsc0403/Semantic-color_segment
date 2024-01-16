import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jinja2 import Environment, FileSystemLoader 
import os
import time

CURRENT_TIME = time.strftime(
    '%Y%m%d_%H_%M_%S', time.localtime(int(time.time())))
print(CURRENT_TIME)

#OUTPUT_PARENT_PATH = rf'E:\Test\Testoutput\{CURRENT_TIME}'
#PALETTE_PARENT_DIR = r"E:\Test\Testoutput\palette_12_12"


def plot_hsv_matrix(data, save_path):
    assert data.shape[-1] == 3
    data_df = pd.DataFrame(data, columns=["hue", "saturation", "value"])
    print("The DataFrame generated from the NumPy array is:")
    sns.set_theme(style="ticks")
    sns.pairplot(data_df)
    plt.savefig(save_path)
    print(data_df)


def batch_plot_hsv_matrix(prompts, save_path):
    for prompt in prompts:
        palette_dir = f"{save_path}/{prompt}"
        hsv_points_list = np.load(rf"{palette_dir}/hsv_points_full_list.npy")
        plot_hsv_matrix(hsv_points_list,
                        save_path=rf"{palette_dir}/full_matrix.png")


def plot_hue(hue_list, save_dir):
    # TODO generate hue ring of every single prompt, rendered with d3.
    # hue belongs to [0, 180]
    os.makedirs(save_dir, exist_ok=True)
    counts, _ = np.histogram(hue_list, range = (0,180), bins=60, density=False)
    counts_string = ""
    for i in range(len(counts)):
        if i == 0:
            counts_string += f"{counts[i]}"
            continue
        counts_string += f",{counts[i]}"
    
    print(counts_string)
    env = Environment(loader=FileSystemLoader(rf'E:\Test\Testcode\semantic-color-code-main\static'))
    template = env.get_template('template.html')    
    with open(f"{save_dir}/hue.html",'w+') as fout:   
        html_content = template.render(data=counts_string)
        fout.write(html_content)



def batch_plot_hue(prompts, src_dir, save_dir):
    for prompt in prompts:
        palette_dir = f"{src_dir}/{prompt}"
        hsv_points_list = np.load(f"{palette_dir}/hsv_points_full_list.npy")
        hue_list = hsv_points_list[:, 0]
        plot_hsv_matrix(data=hsv_points_list, save_path=f"{palette_dir}/hsv_matrix.png")
        plot_hue(hue_list, save_dir= f"{save_dir}/{prompt}")


if __name__ == "__main__":
    prompts = ["apple","banana","buleberry","cherry","grape","peach","tangerine"],
    batch_plot_hsv_matrix(prompts)
