import gzip
from collections import defaultdict
import os
import json
import requests
from PIL import Image
from io import BytesIO
false = False
true = True


def download_image(dataset, item_path, parent_folder="image_text/beauty_image"):
    subfolders = [f.split('.')[0] for f in os.listdir(parent_folder)]
    with open(item_path, 'r') as json_file:
        data = json.load(json_file)
    count = 0

    no_image = []

    for _, one_interaction in data.items():

        if 'imUrl' not in one_interaction:
            print(f'{one_interaction} not image_text')
            no_image.append(one_interaction)
            continue

        count += 1
        asin = one_interaction['asin']

        if asin in subfolders:
            print(f'{count}:file{asin} exist')
            continue

        imUrl = one_interaction['imUrl']
        print(f' from {imUrl} download')
        response = requests.get(imUrl)
        if response.status_code == 200:
            image_data = Image.open(BytesIO(response.content))

            image_path = dataset
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            image_data.save(f'{image_path}/{asin}.png')

            print(f"{count}: image {asin} save in ：f'{dataset}/{asin}")

    print(f' item does not have image：{no_image}')


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def meta_to_5core(core5_dataset='reviews_Beauty/reviews_Beauty_5', meta_dateset='reviews_Beauty/meta_Beauty',
                  save_path='image_text/beauty_image.json'):
    dataname = core5_dataset + '.json.gz'
    meta_dataname = meta_dateset + '.json.gz'

    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    for one_interaction in parse(dataname):
        rev = one_interaction['reviewerID']
        asin = one_interaction['asin']
        countU[rev] += 1
        countP[asin] += 1

    meta_dict = dict()
    for item_meta in parse(meta_dataname):
        meta_dict[item_meta['asin']] = item_meta

    itemmap = dict()
    itemnum = 1
    num = 1
    core5_item_dict = {}

    for one_interaction in parse(dataname):
        rev = one_interaction['reviewerID']
        asin = one_interaction['asin']
        if countU[rev] < 5 or countP[asin] < 5:
            continue

        if asin in itemmap:
            continue
        else:
            num += 1
            itemid = itemnum
            itemmap[asin] = itemid
            itemnum += 1
            core5_item_dict[asin] = meta_dict[asin]

    if not os.path.exists(save_path.split('/')[0]):
        os.makedirs(save_path.split('/')[0])

    with open(save_path, mode='w', encoding='utf-8') as f:
        json.dump(core5_item_dict, f, ensure_ascii=True)


def extract_text(item_path='image_text/beauty.json', save_path='image_text/beauty_text.json', text='title'):

    with open(item_path, 'r') as json_file:
        data = json.load(json_file)

    count = 0
    no_text = []
    item_text = {}
    for _, one_interaction in data.items():

        if text not in one_interaction:
            no_text.append(one_interaction)
            continue

        count += 1
        asin = one_interaction['asin']
        item_text[asin] = one_interaction[text]

        print(f"{count}: get {asin} text ")

    print(f' item does not have text ：{no_text}')

    if not os.path.exists(save_path.split('/')[0]):
        os.makedirs(save_path.split('/')[0])
    with open(save_path, mode='w', encoding='utf-8') as f:
        json.dump(item_text, f, ensure_ascii=True)


if __name__ == '__main__':
    meta_to_5core(core5_dataset='Sports/reviews_Sports_and_Outdoors_5',
                  meta_dateset='Sports/meta_Sports_and_Outdoors',
                  save_path='image_text/Sports_image.json')
    extract_text(item_path='image_text/Sports_image.json',
                 save_path='image_text/Sports_text.json',
                 text='title')
    download_image(dataset='image_text/Sports_image',
                   item_path='image_text/Sports_image.json',
                   parent_folder="image_text/Sports_image")


