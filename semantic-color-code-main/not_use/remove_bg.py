import os
from rembg import remove


def remove_background(image_path, output_path):
    with open(image_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)


def batch_remove_background(prompts, src_img_dir, save_dir):
    if not isinstance(prompts, list):
        prompts = [prompts]

    for prompt in prompts:
        img_dir = rf"{src_img_dir}/{prompt}"
        rembg_dir = rf"{save_dir}/{prompt}"
        os.makedirs(rembg_dir, exist_ok=True)

        for file_name in os.listdir(img_dir):
            image_path = rf"{img_dir}/{file_name}"
            output_path = rf"{rembg_dir}/{file_name}"
            if image_path.endswith(".png"):  # Avoid corner cases
                remove_background(image_path, output_path)


if __name__ == '__main__':
    prompts = ["apple","banana","buleberry","cherry","grape","peach","tangerine"],
    src_img_dir = r"E:\Test\Testoutput\image"
    save_dir = r"E:\Test\Testoutput\image_rembg"
    batch_remove_background(prompts, src_img_dir, save_dir)
