import cv2
import os
import numpy as np
import glob
import supervision as sv
import torch
import torchvision
import time
import sys
sys.path.append('E:\Test\Testcode\GroundSAM\Grounded-Segment-Anything') 
sys.path.append('E:\Test\Testcode\GroundSAM\Grounded-Segment-Anything\segment_anything')
from GroundingDINO.groundingdino.util.inference import Model, predict
from segment_anything import sam_model_registry, SamPredictor

# ...其他代码...
DEVICE = torch.device('cpu')
print("开始了开始了")
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "E:\Test\Testcode\GroundSAM\Grounded-Segment-Anything\GroundingDINO\groundingdino\config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "E:\Test\Testcode\GroundSAM\Grounded-Segment-Anything/groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "E:\Test\Testcode\GroundSAM\Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Predict classes and hyper-param for GroundingDINo
CLASSES = ["horse","sky","grass","tree"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

# 使用检测到的框激活SAM来生成掩码的函数
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    print("到segment了")
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

# 保存分割后的掩码和图像的函数
def save_masked_objects(image, masks, classes, detection_classes, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for class_id in detection_classes:
        # 检查类别 ID 是否在范围内
        if class_id < len(classes):
            class_name = classes[class_id]
            # 为当前类别创建一个与原始图像相同尺寸的透明背景图像
            class_mask_with_alpha = np.zeros((image.shape[0], image.shape[1], 4), dtype=image.dtype)

            # 遍历所有掩码，查找对应当前类别的掩码并应用到透明背景上
            for mask, detected_class_id in zip(masks, detection_classes):
                if detected_class_id == class_id:
                    # 将掩码应用到透明背景
                    mask_alpha = mask * 255  # 将掩码转换为Alpha通道值
                    class_mask_with_alpha[..., :3][mask_alpha > 0] = image[mask_alpha > 0]  # 设置颜色（掩码位置的像素）
                    class_mask_with_alpha[..., 3][mask_alpha > 0] = mask_alpha[mask_alpha > 0]  # 设置Alpha通道（掩码位置的像素）

            # 为了确保文件名的唯一性，使用时间戳
            timestamp = int(time.time())
            class_image_path = os.path.join(output_folder, f"mask_{class_name}_{timestamp}_alpha.png")
            print(f"Saving mask image for class '{class_name}' with alpha: {class_image_path}")
            class_mask_with_alpha_bgr = cv2.cvtColor(class_mask_with_alpha, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(class_image_path, class_mask_with_alpha_bgr)
        else:
            print(f"Warning: Detected class_id {class_id} is out of range. Skipping this mask.")


def process_image(image_file, prompt, save_dir):
    print(f"Processing image: {image_file}")
    image = cv2.imread(image_file)
    if image is None:
        print(f"Failed to read image: {image_file}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"Using prompt: {prompt}")
    # 创建使用 prompt 名称的子目录路径
    output_folder = os.path.join(save_dir, prompt)
    os.makedirs(output_folder, exist_ok=True)

    # 检测对象
    detections = grounding_dino_model.predict_with_classes(
        image=image_rgb,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    # NMS后处理
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")

    # 调用 segment 函数获得掩码
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=image_rgb,
        xyxy=detections.xyxy
    )

    # 调用 save_masked_objects 函数保存结果
    save_masked_objects(
        image=image_rgb,
        masks=detections.mask,
        classes=CLASSES,
        detection_classes=detections.class_id,
        output_folder=output_folder
    )



def batch_segment(prompts, src_dir, save_dir):
    print("开始批量处理图片")

    # 遍历每个提示词，搜索每个提示词对应的子文件夹中的 PNG 图片
    for prompt in prompts:
        # 构建当前提示词对应的子目录路径
        prompt_dir = os.path.join(src_dir, prompt)
        # 搜索当前子目录下的所有 PNG 图片
        image_files = glob.glob(os.path.join(prompt_dir, '*.png'))
        print(f"在{prompt_dir}目录下发现 {len(image_files)} 张图片")

        # 如果当前子目录下存在图片，则对图片进行处理
        if image_files:
            # 遍历找到的所有图像文件并处理
            for image_file in image_files:
                process_image(image_file, prompt, save_dir)
        else:
            print(f"{prompt_dir} 目录下没有找到 PNG 图片。")

    print("批量处理完成")


#if __name__ == "__main__":
    #PROMPTS = ["apple"]  # 示例提示词列表
    #CURRENT_TIME = time.strftime('%Y%m%d_%H_%M_%S', time.localtime(int(time.time())))
    #OUTPUT_PARENT_PATH = rf'E:\Test\Testoutput{CURRENT_TIME}'

    # 创建输出目录
    #os.makedirs(OUTPUT_PARENT_PATH, exist_ok=True)
    #src_dir = rf'{OUTPUT_PARENT_PATH}/image'
    #save_dir = rf'{OUTPUT_PARENT_PATH}/segment'
    #batch_segment(prompts=PROMPTS, src_dir=src_dir, save_dir=save_dir)