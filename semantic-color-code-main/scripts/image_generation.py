import json
import base64
import os
import requests
import time
import random
txt2img_url = 'http://127.0.0.1:7860/sdapi/v1/txt2img'
CKPT_PATH = r'ProAPPs\Git\Gitclone\stable-diffusion-webui\models\Stable-diffusion'
CURRENT_TIME = time.strftime(
    '%Y%m%d_%H_%M_%S', time.localtime(int(time.time())))
OUTPUT_PARENT_PATH = rf'E:\Test\Testoutput{CURRENT_TIME}'

def submit_post(url: str, data: dict):
    """
    Submit a POST request to the given URL with the given data.
    """
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    return response


def save_encoded_image(b64_image: str, output_path: str):
    print(f"Saving image to {output_path}")  # 调试信息
    with open(output_path, 'wb') as image_file:
        image_file.write(base64.b64decode(b64_image))
        
import random

def batch_image_generation(prompts, save_dir, n_samples=5, scales=[5], seeds=[42]):
    # 确保输入是列表
    if not isinstance(scales, list):
        scales = [scales]
    if not isinstance(seeds, list):
        seeds = [seeds]
    if not isinstance(prompts, list):
        prompts = [prompts]

    for prompt in prompts:
        out_dir = rf"{save_dir}/{prompt}"
        os.makedirs(out_dir, exist_ok=True)
        for seed in seeds:
            for scale in scales:
                data = {
                    'prompt': prompt,
                    'n_samples': n_samples,
                    'scale': scale,
                    'seed': seed,
                    'ckpt_path': CKPT_PATH
                }
                

                response = submit_post(txt2img_url, data)
                if response.status_code == 200:
                    response_json = response.json()
                    if 'images' in response_json:  # 假设响应中包含多个图像
                        images_b64 = response_json['images']
                        for i, image_b64 in enumerate(images_b64):
                            output_path = os.path.join(out_dir, f"{prompt}_{scale}_{seed}_{i}.png")
                            save_encoded_image(image_b64, output_path)
                    else:
                        print(f"No images found in response for prompt '{prompt}'")
                else:
                    print(f"Error: {response.status_code} - {response.text}")

if __name__ == '__main__':
    prompts = ["orange","banana","grape"]
    save_dir = rf'{OUTPUT_PARENT_PATH}/image'  # 修改为您的保存目录
    n_samples = 50  # 如果您想为每个种子生成多于一张图片，请增加这个数字
    scales = [5]  # 如果您有多个尺度，可以在列表中添加更多
    seeds = [random.randint(0, 2**32 - 1) for _ in range(10)]  # 生成 4 个随机种子
    batch_image_generation(prompts, save_dir, n_samples, scales, seeds)


