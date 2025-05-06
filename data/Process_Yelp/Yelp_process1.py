# -*- coding: utf-8 -*-
import os
import json
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# -----------------------------
# 1. 参数配置（可根据需要修改）
# -----------------------------
DATASET = "reviews_Yelp"          # 处理后的数据目录名
REVIEW_FILE = "/root/Yelp/yelp_dataset/yelp_academic_dataset_review.json"  # Yelp 评论文件（JSON格式）
MIN_INTERACTIONS = 5             # 用户与商家均需要 >= 5 条交互才保留

# 生成的文件路径
train_file = f"./{DATASET}/train.txt"
valid_file = f"./{DATASET}/valid.txt"
test_file  = f"./{DATASET}/test.txt"
imap_file  = f"./{DATASET}/imap.json"
umap_file  = f"./{DATASET}/umap.json"
data_file  = f"./{DATASET}.txt"

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
      ...
    }
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 排除空行
                yield json.loads(line)

# -----------------------------
# 4. 第一次遍历：统计用户和商家交互次数
# -----------------------------
countU = defaultdict(int)  # 用户交互数
countP = defaultdict(int)  # 商家交互数

print("Counting interactions...")
for one_interaction in tqdm(parse_yelp_reviews(REVIEW_FILE)):
    user_id = one_interaction["user_id"]
    business_id = one_interaction["business_id"]
    countU[user_id] += 1
    countP[business_id] += 1

# -----------------------------
# 5. 第二次遍历：构建 User 字典，并过滤交互数不足的用户/商家
# -----------------------------
usermap = dict()  # 原始 user_id => 新的整数 ID
itemmap = dict()  # 原始 business_id => 新的整数 ID
usernum = 1
itemnum = 1

User = dict()  # { new_user_id: [[new_item_id, timestamp], ...], ... }

print("Building user-item interactions...")
for one_interaction in tqdm(parse_yelp_reviews(REVIEW_FILE)):
    user_id = one_interaction["user_id"]
    business_id = one_interaction["business_id"]
    
    # 如果交互数不足，则跳过
    if countU[user_id] < MIN_INTERACTIONS or countP[business_id] < MIN_INTERACTIONS:
        continue
    
    # 将日期转换为时间戳（也可以按需要保留原始 YYYY-MM-DD）
    date_str = one_interaction["date"]  # e.g. "2016-03-09"
    # 将日期转换为一个 float/int，便于排序
    # 格式化成 unix timestamp (可选方案)
    # timestamp = datetime.strptime(date_str, "%Y-%m-%d").timestamp()
    timestamp = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp()

    # user_id => new_user_id
    if user_id not in usermap:
        usermap[user_id] = usernum
        User[usernum] = []
        usernum += 1
    
    # business_id => new_item_id
    if business_id not in itemmap:
        itemmap[business_id] = itemnum
        itemnum += 1
    
    new_user_id = usermap[user_id]
    new_item_id = itemmap[business_id]
    
    # 保存记录 [new_item_id, timestamp]
    User[new_user_id].append([new_item_id, timestamp])

# -----------------------------
# 6. 保存映射表
# -----------------------------
with open(imap_file, 'w', encoding='utf-8') as f:
    json.dump(itemmap, f)

with open(umap_file, 'w', encoding='utf-8') as f:
    json.dump(usermap, f)

# -----------------------------
# 7. 对每个用户的交互记录按时间排序
# -----------------------------
for uid in User:
    User[uid].sort(key=lambda x: x[1])

# -----------------------------
# 8. 划分训练、验证和测试集
# -----------------------------
user_train = {}
user_valid = {}
user_test  = {}

for uid in User:
    interactions = User[uid]
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
    with open(outfile, 'w') as f:
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

print("Done.")
