import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmcv import Config, DictAction
from terminaltables import AsciiTable
# from pycocotools.coco import COCO
from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from mmdet.models import build_detector
from .custom import CustomDataset
from PIL import Image, ImageDraw, ImageFont
import copy
import random
import json
import matplotlib.pyplot as plt
from collections import Counter
import os
from PIL import Image

import json

def extract_category_ids_above_threshold(dictionary, threshold):
    result = [key for key, value in dictionary.items() if value >= threshold]
    return result

def count_category_ids(coco_json_file):
    with open(coco_json_file, 'r') as f:
        data = json.load(f)

    category_counts = {}

    for annotation in data['annotations']:
        category_id = annotation['category_id']
        if category_id in category_counts:
            category_counts[category_id] += 1
        else:
            category_counts[category_id] = 1

    return category_counts

def crop_image(annotation):
    #print(annotation)
    root = "/large/ttani_2/bhrl/data/manga_dataset/images/"
    image_file = annotation['file_name']
    image_category = annotation["ann"]["category_id"]
    image = Image.open(root + image_file)

    ann_data = annotation['ann']
    bbox = ann_data['bbox']
    cropped_image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
    
    output_folder = f"/large/ttani_2/bhrl/data/cropped_manga/{image_file}"
    
    os.makedirs(output_folder, exist_ok=True)  # フォルダが存在しない場合に作成する
    cropped_image.save(f"{output_folder}/{image_category}.jpg")

def load_coco_categories(annotation_file_path):
    with open(annotation_file_path, 'r') as f:
        data = json.load(f)
    
    categories = tuple(category['name'] for category in data['categories'])
    
    return categories

def find_indexes(lst, arr):
    indexes = []
    for item in lst:
        if item in arr:
            indexes.append(arr.index(item))
    return indexes

def remove_elements(source_list, elements_to_remove):
    return [item for item in source_list if item not in elements_to_remove]

def plot_histogram_from_dict(dictionary, filename):
    # 辞書を値で降順にソート
    sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))

    # ソートされた辞書からキーと値を取り出す
    keys = range(len(sorted_dict))
    values = sorted_dict.values()

    # ヒストグラムを作成
    plt.bar(keys,values)
    plt.ylabel("Number of appearances")
    plt.xlabel("Character (sorted by most)")

    # ヒストグラムを画像として保存
    plt.savefig(filename)

# サンプルの辞書で関数をテスト


@DATASETS.register_module()
class OneShotVOCDataset(CustomDataset):



    def __init__(self,
                 ann_file,
                 pipeline,
                 config_path,
                 threshold=0,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 test_seen_classes=False,
                 position=0,
                 ):
        self.config_path =config_path
        cfg = mmcv.Config.fromfile(config_path)
        CLASSES = list(set(load_coco_categories(cfg.data.train.ann_file) +load_coco_categories(cfg.data.test.ann_file)))
        train_seen_classes_str = list(set(load_coco_categories(cfg.data.train.ann_file)))
        train_seen_classes= find_indexes(CLASSES,train_seen_classes_str)
        counts1 = count_category_ids(cfg.data.test.ann_file)
        self.split = train_seen_classes

        self.test_seen_classes = test_seen_classes
        self.position = position
        self.selected_category_for_test = extract_category_ids_above_threshold(counts1,threshold)
        self.train_seen_classes = train_seen_classes

        classes = CLASSES 
        super(OneShotVOCDataset,
              self).__init__(ann_file, pipeline, classes, data_root, img_prefix,
                             seg_prefix, proposal_file, test_mode)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.split_cats(self.train_seen_classes)
        self.img_ids = self.coco.get_img_ids()
        img_infos, img_cates = self.generate_infos()
        self.cates = img_cates
        #print(self.cates)
        return img_infos

    def split_cats(self,train_seen_classes):
        self.train_cat =[]
        self.test_cat = []

        #print(self.cat_ids)
        for i in range(len(self.cat_ids)):
            if i not in self.split and i in self.selected_category_for_test:
                self.test_cat.append(self.cat_ids[i])
            else:
                self.train_cat.append(self.cat_ids[i])
        if self.test_seen_classes:
            test_seen_cat = [x for x in train_seen_classes if x in self.selected_category_for_test]
            self.test_cat = test_seen_cat
        print(self.test_cat)

    def generate_infos(self):
        img_infos = []
        img_cates = []
        for i in self.img_ids:
            if not self.test_mode:
                img_infos, img_cates = self.generate_train(i, img_infos) 
            else:
                img_infos, img_cates = self.generate_test(i, img_infos, img_cates)
        return img_infos, img_cates

    def generate_train(self, i, img_infos):
        info = self.coco.load_imgs([i])[0]
        info['filename'] = info['file_name']
        img_anns_ids = self.coco.get_ann_ids(img_ids=[i])
        img_anns = self.coco.load_anns(img_anns_ids)
        #print("This is seen classes to train")
        #print(img_anns)
        for img_ann in img_anns:
            if img_ann['category_id'] in self.train_cat:
                img_infos.append(info)
                break
        return img_infos, None

    def generate_test(self, i, img_infos, img_cates):
        info = self.coco.loadImgs([i])[0]
        #print(info)
        info['filename'] = info['file_name']
        img_anns_ids = self.coco.getAnnIds(imgIds=i)
        #print(img_anns_ids)
        img_anns = self.coco.loadAnns(img_anns_ids)
        img_cats = list()
        for img_ann in img_anns:
            if img_ann['category_id'] in img_cats:
                continue
            elif img_ann['category_id'] in self.test_cat:
                img_cats.append(img_ann['category_id'])
                img_infos.append(info)
                img_cates.append(img_ann['category_id'])
            else:
                continue
        #print()
        return img_infos, img_cates

    def get_ann_info(self, idx, cate=None):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids) 
        if cate is None: 
            cate = self.random_cate(ann_info)
        return self._parse_ann_info(self.data_infos[idx], ann_info, cate)

    def _filter_imgs(self, min_size=32):#original 32
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        ids_in_cat &= ids_with_ann
        
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def random_cate(self, ann_info):
        index = np.random.randint(len(ann_info))
        cate = ann_info[index]['category_id']

        if not self.test_mode:
            cates = self.train_cat
        else:
            cates = self.test_cat

        while cate not in cates:
            index = np.random.randint(len(ann_info))
            cate = ann_info[index]['category_id']
        #print(cate)
        return cate

    def _parse_ann_info(self, img_info, ann_info, cate_select):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            elif ann['category_id'] == cate_select:
                gt_bboxes.append(bbox)
                gt_labels.append(0)
                #gt_masks_ann.append(ann['segmentation'])
            else:
                continue

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(bboxes=gt_bboxes,
                   labels=gt_labels,
                   bboxes_ignore=gt_bboxes_ignore,
                   masks=gt_masks_ann,
                   seg_map=seg_map,
                   cate=cate_select)

        return ann

    def prepare_train_ref_img(self, idx, cate):
        rf_img_info = dict()
        img_id = self.data_infos[idx]['id']
        rf_ids = self.coco.getImgIds(catIds=[cate])
        while True:
            rf_id = rf_ids[np.random.randint(0, len(rf_ids))]
            while rf_id == img_id and len(rf_ids) > 1:
                rf_id = rf_ids[np.random.randint(0, len(rf_ids))]
            rf_anns = self.coco.loadAnns(
                self.coco.getAnnIds(imgIds=rf_id, catIds=cate, iscrowd=False))
            if len(rf_anns) > 0:
                rand_index = np.random.randint(len(rf_anns))
                rf_img_info['ann'] = rf_anns[rand_index]
                rf_img_info['file_name'] = self.coco.loadImgs([rf_id])[0]['file_name']
                break
        rf_img_info['img_info'] = self.coco.loadImgs([rf_id])[0]
        #print(rf_img_info)
        #print("this is train classes")
        #print(rf_img_info)
        crop_image(rf_img_info)
        return rf_img_info

    def prepare_test_ref_img(self, idx, cate):
        rf_img_info = dict()
        img_id = self.data_infos[idx]['id']
        #print(img_id)
        rf_ids = self.coco.getAnnIds(catIds=[cate], iscrowd=False)
        
        #print(rf_ids)
        
        random.seed(img_id)
        l = list(range(len(rf_ids)))
        #print
        #random.shuffle(l)
        #print(l[0])
        #print(l)

        position = l[(self.position) % len(l)]

        #print(position)
        #print(position)
        #print(position)
        ref = rf_ids[position]
        #print(rf_ids)
        #print(ref)
        #print(position)

        rf_anns = self.coco.loadAnns(ref)[0]

        #print(rf_anns)
        rf_img_info['ann'] = rf_anns
        rf_img_info['file_name'] = self.coco.loadImgs(rf_anns['image_id'])[0]['file_name']
        rf_img_info['img_info'] = self.coco.loadImgs(rf_anns['image_id'])[0]
        #print(rf_img_info)
        crop_image(rf_img_info)
        return rf_img_info

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        rf_img_info = self.prepare_train_ref_img(idx, ann_info['cate'])
        #print(ann_info['cate'])
        results = dict(img_info=img_info,
                       ann_info=ann_info,
                       rf_img_info=rf_img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx, self.cates[idx])
        #print(idx)
     
        rf_img_info = self.prepare_test_ref_img(idx, self.cates[idx])

        #print(ann_info["ann"]["category_id"])
        #print(rf_img_info)
        results = dict(img_info=img_info,
                       ann_info=ann_info,
                       rf_img_info=rf_img_info,
                       label=self.cat2label[self.cates[idx]])
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
            #print(results["proposals"])

        self.pre_pipeline(results)
        crop_image(rf_img_info)
        #print(str(self.pre_pipeline(results)))
        #print(rf_img_info)
        return self.pipeline(results)
