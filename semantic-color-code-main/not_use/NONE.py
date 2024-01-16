import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
import os
import time 
import numpy as np
import supervision as sv
import glob
from GroundingDINO.groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

CURRENT_TIME = time.strftime('%Y%m%d_%H_%M_%S', time.localtime(int(time.time())))
print(CURRENT_TIME)
OUTPUT_PARENT_PATH = rf'E:\Test\Testoutput{CURRENT_TIME}'
src_dir = rf'{OUTPUT_PARENT_PATH}/image'
save_dir = rf'{OUTPUT_PARENT_PATH}/segment'

DEVICE = 'cpu'

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

# Predict classes and hyper-param for GroundingDINO
SOURCE_IMAGE_PATH = rf'{OUTPUT_PARENT_PATH}/image'
CLASSES = ["Horse", "Clouds", "Grasses", "Sky", "Hill"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

# load image
image = cv2.imread(SOURCE_IMAGE_PATH)

# detect objects
detections = grounding_dino_model.predict_with_classes(
    image=image,
    classes=CLASSES,
    box_threshold=BOX_THRESHOLD,
    text_threshold=BOX_THRESHOLD
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _ 
    in detections]

annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

# NMS post process
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

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


# convert detections to masks
detections.mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    xyxy=detections.xyxy
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _ 
    in detections]
annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

def save_masked_objects(image, masks, classes, detection_classes, output_folder="extracted_objects"):
    os.makedirs(output_folder, exist_ok=True)
    for index, (mask, class_id) in enumerate(zip(masks, detection_classes)):
        # 获取类名
        class_name = classes[class_id]

        # 获取掩码对应的区域
        y_indices, x_indices = np.where(mask > 0)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # 裁剪掩码和图像区域
        object_mask = mask[y_min:y_max+1, x_min:x_max+1]
        object_image = image[y_min:y_max+1, x_min:x_max+1]

        # 确保使用RGB颜色空间
        object_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)

        # 创建一个具有Alpha通道的图像
        object_with_alpha = np.zeros((object_mask.shape[0], object_mask.shape[1], 4), dtype=object_image.dtype)

        # 使用掩码设置对象图像的颜色
        object_with_alpha[..., :3] = object_image  # 设置颜色
        object_with_alpha[..., 3] = object_mask * 255  # 设置alpha通道

        # 保存图像
        output_path = os.path.join(output_folder, f"object_{index}_{class_name}.png")
        cv2.imwrite(output_path, object_with_alpha)
        print(f"Saved: {output_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

output_folder = save_dir
os.makedirs(output_folder, exist_ok=True)

save_masked_objects(
    image=image_rgb, 
    masks=detections.mask,
    classes=CLASSES,
    detection_classes=detections.class_id,
    output_folder=output_folder  # 使用之前定义的包含当前时间的文件夹路径
)

def batch_segemnt(prompts, src_img_dir, save_path):
    for prompt in prompts:
        src_img_path = os.path.join(src_img_dir, prompt)
        save_path = os.path.join(save_dir, prompt)
        os.makedirs(save_path, exist_ok=True)
        
        
# 指定基于当前时间的文件夹路径
output_folder_dino = save_dir
output_folder_sam =save_dir
os.makedirs(output_folder_dino, exist_ok=True)
os.makedirs(output_folder_sam, exist_ok=True)

# 创建文件名，并将它们与文件夹路径合并以形成完整的文件路径
annotated_dino_image_path = os.path.join(output_folder_dino, f"groundingdino_annotated_image_{CURRENT_TIME}.jpg")
annotated_sam_image_path = os.path.join(output_folder_sam, f"groundedsam_annotated_image_{CURRENT_TIME}.jpg")

# 使用完整路径保存图片
cv2.imwrite(annotated_dino_image_path, annotated_frame)
cv2.imwrite(annotated_sam_image_path, annotated_image)

