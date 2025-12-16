# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = 'datasets'
    DATASETS = {
        'coco_2017_train': {
            'img_dir': 'coco/train2017',
            'ann_file': 'coco/annotations/instances_train2017.json'
        },
        'coco_2017_val': {
            'img_dir': 'coco/val2017',
            'ann_file': 'coco/annotations/instances_val2017.json'
        },
        'coco_2017_test_dev': {
            'img_dir': 'coco/test2017',
            'ann_file': 'coco/annotations/image_info_test-dev2017.json'
        },
        'custom_train_coco': {
            'img_dir': 'custom_dataset/images',
            'ann_file': 'custom_dataset/annotations/train.json'
        },
        'custom_val_coco': {
            'img_dir': 'custom_dataset/images',
            'ann_file': 'custom_dataset/annotations/val.json'
        },
        '2class_train_coco': {
            'img_dir': 'dataset_2class/images',
            'ann_file': 'dataset_2class/annotations/train.json'
        },
        '2class_val_coco': {
            'img_dir': 'dataset_2class/images',
            'ann_file': 'dataset_2class/annotations/val.json'
        },
        '2class_fixed_train_coco': {
            'img_dir': 'dataset_2class_fixed/images',
            'ann_file': 'dataset_2class_fixed/annotations/train.json'
        },
        '2class_fixed_val_coco': {
            'img_dir': 'dataset_2class_fixed/images',
            'ann_file': 'dataset_2class_fixed/annotations/val.json'
        },
        'S_train_coco': {
            'img_dir': 'dataset_S_coco/images',
            'ann_file': 'dataset_S_coco/annotations/train.json'
        },
        'S_val_coco': {
            'img_dir': 'dataset_S_coco/images',
            'ann_file': 'dataset_S_coco/annotations/val.json'
        },
        'L_train_coco': {
            'img_dir': 'dataset_L_coco/images',
            'ann_file': 'dataset_L_coco/annotations/train.json'
        },
        'L_val_coco': {
            'img_dir': 'dataset_L_coco/images',
            'ann_file': 'dataset_L_coco/annotations/val.json'
        },
        'S_norot_train_coco': {
            'img_dir': 'dataset_S_coco_norot/images',
            'ann_file': 'dataset_S_coco_norot/annotations/train.json'
        },
        'S_norot_val_coco': {
            'img_dir': 'dataset_S_coco_norot/images',
            'ann_file': 'dataset_S_coco_norot/annotations/val.json'
        },
        'L_norot_train_coco': {
            'img_dir': 'dataset_L_coco_norot/images',
            'ann_file': 'dataset_L_coco_norot/annotations/train.json'
        },
        'L_norot_val_coco': {
            'img_dir': 'dataset_L_coco_norot/images',
            'ann_file': 'dataset_L_coco_norot/annotations/val.json'
        },
        }

    @staticmethod
    def get(name):
        if 'coco' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs['img_dir']),
                ann_file=os.path.join(data_dir, attrs['ann_file']),
            )
            return dict(
                factory='COCODataset',
                args=args,
            )
        else:
            raise RuntimeError('Only support coco format now!')
        return None
