# -*- coding: utf-8 -*-
import os
import json
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# -----------------------------
# 1. 参数配置（可根据需要修改）
# -----------------------------
DATASET = "reviews_Yelp_new"  # 处理后的数据目录名

# Yelp 评论文件（JSON格式），包含 user_id、business_id、text、date 等
REVIEW_FILE = "/root/Yelp/yelp_dataset/yelp_academic_dataset_review.json"

# Yelp 图片文件（JSON格式），包含 photo_id、business_id、caption、label 等
PHOTO_FILE = "/root/Yelp/yelp_photos/photos.json"

MIN_INTERACTIONS = 5  # 用户与商家均需要 >= 5 条交互才保留

# 生成的文件路径
train_file = f"./{DATASET}/train.txt"
valid_file = f"./{DATASET}/valid.txt"
test_file = f"./{DATASET}/test.txt"
imap_file = f"./{DATASET}/imap.json"
umap_file = f"./{DATASET}/umap.json"
data_file = f"./{DATASET}.txt"

# 额外输出：文本和图像信息
text_map_file = f"./{DATASET}/text.json"   # { (new_user_id, new_item_id): [list_of_review_texts...] }
photo_map_file = f"./{DATASET}/photos.json" # { new_item_id: [ {photo_id, caption, label}, ... ] }

# -----------------------------
# 2. 创建输出目录
# -----------------------------
if not os.path.isdir(f"./{DATASET}"):
    os.mkdir(f"./{DATASET}")

# -----------------------------
# 3. 辅助函数：解析 Yelp 评论
# -----------------------------
def parse_yelp_reviews(path):
    """
    逐行读取 Yelp JSON 文件，每行是一个 review 对象：
    {
      "review_id": ...,
      "user_id": ...,
      "business_id": ...,
      "stars": ...,
      "date": "YYYY-MM-DD",
      "text": "...",
      ...
    }
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# -----------------------------
# 4. 解析 photo.json
# -----------------------------
def parse_yelp_photos(path):
    """
    逐行读取 Yelp photo.json，每行是一个 photo 对象：
    {
       "photo_id"    : "...",
       "business_id" : "...",
       "caption"     : "...",
       "label"       : "food" / "drink" / ...
    }
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# -----------------------------
# 5. 第一次遍历：统计用户和商家交互次数
# -----------------------------
countU = defaultdict(int)  # 用户交互数
countP = defaultdict(int)  # 商家交互数

print("Counting interactions...")
for one_interaction in tqdm(parse_yelp_reviews(REVIEW_FILE), desc="Reading reviews"):
    user_id = one_interaction["user_id"]
    business_id = one_interaction["business_id"]
    countU[user_id] += 1
    countP[business_id] += 1

# -----------------------------
# 6. 第二次遍历：构建 User 字典，并过滤交互数不足的用户/商家
# -----------------------------
usermap = dict()   # 原始 user_id => 新的整数 ID
itemmap = dict()   # 原始 business_id => 新的整数 ID
usernum = 1
itemnum = 1

# 存放用户交互： { new_user_id: [[new_item_id, timestamp], ...], ... }
User = defaultdict(list)

# 存放用户文本： { (new_user_id, new_item_id): [list_of_review_texts], ... }
#  - 一个 (user, item) 可能有多条 review，所以用 list
TextData = defaultdict(list)

print("Building user-item interactions & collecting text...")
for one_interaction in tqdm(parse_yelp_reviews(REVIEW_FILE), desc="Filtering & Mapping"):
    user_id = one_interaction["user_id"]
    business_id = one_interaction["business_id"]
    date_str = one_interaction["date"]  # e.g. "2016-03-09"
    review_text = one_interaction.get("text", "")
    
    # 如果交互数不足，则跳过
    if countU[user_id] < MIN_INTERACTIONS or countP[business_id] < MIN_INTERACTIONS:
        continue

    # 处理时间：拼接成 "%Y-%m-%d HH:MM:SS" 如果只有日期
    # Yelp 数据一般是 "YYYY-MM-DD"；若不含时间则补 "00:00:00"
    if len(date_str) == 10:  # e.g. "2016-03-09"
        date_str += " 00:00:00"

    # 转换为时间戳
    timestamp = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp()

    # user_id => new_user_id
    if user_id not in usermap:
        usermap[user_id] = usernum
        usernum += 1

    # business_id => new_item_id
    if business_id not in itemmap:
        itemmap[business_id] = itemnum
        itemnum += 1

    new_user_id = usermap[user_id]
    new_item_id = itemmap[business_id]

    # 保存交互记录 [new_item_id, timestamp]
    User[new_user_id].append([new_item_id, timestamp])

    # 保存文本 - 同一个 (user, item) 可能多条评论，追加
    TextData[(new_user_id, new_item_id)].append(review_text)

# -----------------------------
# 7. 保存映射表
# -----------------------------
with open(imap_file, 'w', encoding='utf-8') as f:
    json.dump(itemmap, f)

with open(umap_file, 'w', encoding='utf-8') as f:
    json.dump(usermap, f)

# 对每个用户的交互记录按时间排序
for uid in User:
    User[uid].sort(key=lambda x: x[1])

# -----------------------------
# 8. 划分训练、验证和测试集
# -----------------------------
user_train = {}
user_valid = {}
user_test  = {}

for uid, interactions in User.items():
    n = len(interactions)
    if n < 3:
        # 若不足 3 条，则全部归为训练
        user_train[uid] = interactions
        user_valid[uid] = []
        user_test[uid]  = []
    else:
        user_train[uid] = interactions[:-2]
        user_valid[uid] = [interactions[-2]]
        user_test[uid]  = [interactions[-1]]

# -----------------------------
# 9. 打印一些统计信息
# -----------------------------
print(f"Total users (new IDs): {usernum - 1}")
print(f"Total items (new IDs): {itemnum - 1}")

num_instances = sum(len(User[u]) for u in User)
print("total user: ", len(User))
print("total instances: ", num_instances)
print("avg length: ", num_instances / len(User) if len(User) > 0 else 0)
print("total items: ", itemnum - 1)

density = 0
if len(User) > 0 and (itemnum - 1) > 0:
    density = num_instances / (len(User) * (itemnum - 1))
print("density: ", density)

numvalid_instances = sum(len(v) for v in user_valid.values())
numtest_instances  = sum(len(t) for t in user_test.values())
print('valid #users: ', len(user_valid))
print('valid instances: ', numvalid_instances)
print('test #users: ', len(user_test))
print('test instances: ', numtest_instances)

# -----------------------------
# 10. 写文件的函数
# -----------------------------
def write_file_v2(data_dict, outfile):
    """
    写入格式：
    userID  item1 item2 ...
    每行一个用户
    """
    with open(outfile, 'w', encoding='utf-8') as f:
        for u, ilist in sorted(data_dict.items()):
            f.write(str(u))
            for i, t in ilist:
                f.write(f" {i}")
            f.write("\n")

# -----------------------------
# 11. 输出交互数据
# -----------------------------
write_file_v2(User, data_file)
print(f"All data interactions saved to {data_file}")

# 如果需要分别写 train/valid/test，可以调用类似函数
write_file_v2(user_train, train_file)
write_file_v2(user_valid, valid_file)
write_file_v2(user_test,  test_file)

# -----------------------------
# 12. 处理多模态：文本 (review) + 图像 (photo)
# -----------------------------
# 12.1 保存文本信息：{(uid, iid): ["review1", "review2", ...], ...}
text_map = {}
for (uid, iid), texts in TextData.items():
    # 可以将多条评论合并为一个长字符串，也可以直接保存 list
    # 这里简单示例：合并为一个字符串
    merged_text = " ".join(texts)
    text_map[f"{uid}-{iid}"] = merged_text

with open(text_map_file, 'w', encoding='utf-8') as f:
    json.dump(text_map, f, ensure_ascii=False)
print(f"Text data saved to {text_map_file}")

# 12.2 解析 photo.json，关联到 itemmap
photo_map = defaultdict(list)  # { new_item_id: [ {photo_id, caption, label}, ... ] }
for photo_info in tqdm(parse_yelp_photos(PHOTO_FILE), desc="Parsing photo.json"):
    biz_id = photo_info["business_id"]
    if biz_id in itemmap:
        new_item_id = itemmap[biz_id]
        entry = {
            "photo_id": photo_info["photo_id"],
            "caption": photo_info.get("caption", ""),
            "label": photo_info.get("label", "")
        }
        photo_map[new_item_id].append(entry)

# 写出 {item_id: [photo_data,...]}
photo_map_dict = {}
for iid, plist in photo_map.items():
    photo_map_dict[str(iid)] = plist  # 转为 str key 便于 json 序列化

with open(photo_map_file, 'w', encoding='utf-8') as f:
    json.dump(photo_map_dict, f, ensure_ascii=False)

print(f"Photo data saved to {photo_map_file}")

print("Done (Multimodal data prepared).")
