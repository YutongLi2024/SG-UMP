import os
import tarfile
import json
import csv
import collections
import argparse
from pathlib import Path
from tqdm import tqdm  # 添加 tqdm 用于显示进度条


def extract_tar(tar_path, extract_path):
    """解压 tar 文件到指定路径"""
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        for member in tqdm(tar.getmembers(), desc=f"Extracting {tar_path}", unit="file"):
            tar.extract(member, path=extract_path)
    print(f"Extracted {tar_path} to {extract_path}")


def json_to_csv(json_file_path, csv_file_path):
    """将 JSON 文件转换为 CSV 文件"""
    def get_column_names(data, parent_key=''):
        """递归提取 JSON 数据中的列名（支持嵌套字段）"""
        column_names = []
        for k, v in data.items():
            column_name = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, collections.abc.MutableMapping):
                column_names.extend(get_column_names(v, column_name))
            else:
                column_names.append(column_name)
        return column_names

    def get_nested_value(data, key):
        """通过嵌套键获取 JSON 值"""
        keys = key.split('.')
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return None
        return data

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = [json.loads(line) for line in tqdm(json_file, desc=f"Reading {json_file_path}", unit="line")]
    
    # 获取所有列名
    column_names = set()
    for entry in tqdm(data, desc="Extracting column names", unit="row"):
        column_names.update(get_column_names(entry))
    column_names = sorted(column_names)

    # 写入 CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(column_names)
        for entry in tqdm(data, desc=f"Writing to {csv_file_path}", unit="row"):
            row = [get_nested_value(entry, col) for col in column_names]
            writer.writerow(row)

    print(f"Converted {json_file_path} to {csv_file_path}")


def process_yelp_photos(photo_dir, output_dir):
    """处理 Yelp 图片数据（例如分类或重命名）"""
    photo_dir = Path(photo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for photo_file in tqdm(list(photo_dir.iterdir()), desc="Processing photos", unit="photo"):
        if photo_file.suffix in ['.jpg', '.jpeg', '.png']:
            new_name = f"{photo_file.stem}_processed{photo_file.suffix}"
            photo_file.rename(output_dir / new_name)
    print(f"Processed photos in {photo_dir} to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Process Yelp dataset")
    parser.add_argument('--yelp_tar', type=str, required=True, help='Path to yelp_dataset.tar file')
    parser.add_argument('--photos_tar', type=str, required=True, help='Path to yelp_photos.tar file')
    parser.add_argument('--output_dir', type=str, default='Yelp', help='Path to output directory (default: Yelp)')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解压 Yelp 数据集
    yelp_dataset_dir = output_dir / "yelp_dataset"
    yelp_photos_dir = output_dir / "yelp_photos"
    yelp_dataset_dir.mkdir(exist_ok=True)
    yelp_photos_dir.mkdir(exist_ok=True)

    extract_tar(args.yelp_tar, yelp_dataset_dir)
    extract_tar(args.photos_tar, yelp_photos_dir)

    # 查找所有 JSON 文件并转换为 CSV
    for json_file in yelp_dataset_dir.glob("*.json"):
        csv_file_path = output_dir / f"{json_file.stem}.csv"
        json_to_csv(json_file, csv_file_path)

    # 处理图片
    processed_photos_dir = output_dir / "processed_photos"
    process_yelp_photos(yelp_photos_dir, processed_photos_dir)


if __name__ == "__main__":
    main()
