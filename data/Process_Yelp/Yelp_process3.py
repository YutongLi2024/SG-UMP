# -*- coding: utf-8 -*-
import os
import json
from tqdm import tqdm
import torch
import clip
from PIL import Image

# -----------------------------
# 1. 参数配置
# -----------------------------
DATASET = "reviews_Yelp_new"  # 数据集名称
TEXT_MAP_FILE = f"./{DATASET}/text.json"   # 文本映射文件
PHOTO_MAP_FILE = f"./{DATASET}/photos.json"  # 图像映射文件
IMAGE_BASE_PATH = f"/root/Yelp/yelp_photos/photos"  # 图像存储目录

# 保存特征的路径
TEXT_FEATURE_FILE = f"./{DATASET}/clip_text_features_new.pt"
IMAGE_FEATURE_FILE = f"./{DATASET}/clip_image_features_new.pt"

# 检查设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2. 加载 CLIP 模型
# -----------------------------
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()  # 设为评估模式

# -----------------------------
# 3. 加载文本和图像数据
# -----------------------------
# 加载文本映射
with open(TEXT_MAP_FILE, "r", encoding="utf-8") as f:
    text_map = json.load(f)
print(f"Loaded {len(text_map)} text entries.")

# 加载图像映射
with open(PHOTO_MAP_FILE, "r", encoding="utf-8") as f:
    photo_map = json.load(f)
print(f"Loaded {len(photo_map)} photo entries.")

# -----------------------------
# 4. 提取特征：确保与 itemmap 对齐
# -----------------------------
# 初始化特征字典
text_features = {}
image_features = {}

# 初始化默认特征
default_text_feature = torch.zeros(1, 512).to(device)  # CLIP 的文本输出向量维度为 512
default_image_feature = torch.zeros(1, 512).to(device)  # CLIP 的图像输出向量维度为 512

imap_file = f"/root/reviews_Yelp_new/imap.json"
with open(imap_file, 'r', encoding='utf-8') as f:
    itemmap = json.load(f)

# 提取特征
for item_id in tqdm(itemmap.values(), desc="Processing items"):
    # 1. 处理文本特征
    if str(item_id) in text_map:
        text = text_map[str(item_id)]
        tokenized_text = clip.tokenize(text, truncate=True).to(device)
        with torch.no_grad():
            text_feature = model.encode_text(tokenized_text)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)  # 归一化
        text_features[item_id] = text_feature.cpu().numpy()  # 保存到字典中
    else:
        # 如果没有文本，使用默认值
        text_features[item_id] = default_text_feature.cpu().numpy()

    # 2. 处理图像特征
    if str(item_id) in photo_map:
        photos = photo_map[str(item_id)]
        image_feature_list = []
        for photo in photos:
            photo_id = photo["photo_id"]
            image_path = os.path.join(IMAGE_BASE_PATH, f"{photo_id}.jpg")

            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            try:
                # 提取单张图像特征
                image = Image.open(image_path).convert("RGB")
                preprocessed_image = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    single_image_feature = model.encode_image(preprocessed_image)
                    single_image_feature /= single_image_feature.norm(dim=-1, keepdim=True)  # 归一化
                image_feature_list.append(single_image_feature)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

        # 如果有多张图片，取平均特征
        if image_feature_list:
            image_feature = torch.mean(torch.stack(image_feature_list), dim=0)
            image_features[item_id] = image_feature.cpu().numpy()
        else:
            # 如果没有有效图像，使用默认值
            image_features[item_id] = default_image_feature.cpu().numpy()
    else:
        # 如果没有图像，使用默认值
        image_features[item_id] = default_image_feature.cpu().numpy()

# -----------------------------
# 5. 保存特征
# -----------------------------
# 保存文本特征
torch.save(text_features, TEXT_FEATURE_FILE)
print(f"Text features saved to {TEXT_FEATURE_FILE}.")

# 保存图像特征
torch.save(image_features, IMAGE_FEATURE_FILE)
print(f"Image features saved to {IMAGE_FEATURE_FILE}.")

