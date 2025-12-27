#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 5
        self.miscs.ckpt_interval_epochs = 25
        self.miscs.print_interval_iters = 6  # 1エポックに1回ログ出力

        # Total epochs
        self.train.total_epochs = 200

        # Pretrained model path for finetuning
        self.train.finetune_path = './pretrained_models/damoyolo_tinynasL20_T.pth'

        # optimizer
        self.train.batch_size = 8
        self.train.base_lr_per_img = 0.01 / 64
        self.train.min_lr_ratio = 0.05
        self.train.weight_decay = 5e-4
        self.train.momentum = 0.9
        self.train.no_aug_epochs = 30
        self.train.warmup_epochs = 3

        # augment - 強化版（±90度回転 + flip）
        self.train.augment.transform.image_max_range = (640, 640)
        self.train.augment.transform.flip_prob = 0.5  # 水平反転50%
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 90.0  # ±90度回転
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 0.2
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)

        # Dataset L (no rotation)
        self.dataset.train_ann = ('L_norot_train_coco', )
        self.dataset.val_ann = ('L_norot_val_coco', )

        # backbone
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_L20_k1kx.txt')
        TinyNAS = {
            'name': 'TinyNAS_res',
            'net_structure_str': structure,
            'out_indices': (2, 4, 5),
            'with_spp': True,
            'use_focus': True,
            'act': 'relu',
            'reparam': True,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 1.0,
            'hidden_ratio': 1.0,
            'in_channels': [96, 192, 384],
            'out_channels': [64, 128, 256],
            'act': 'relu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
        }

        self.model.neck = GiraffeNeckV2

        # Single class detection (L class)
        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': 1,
            'in_channels': [64, 128, 256],
            'stacked_convs': 0,
            'reg_max': 16,
            'act': 'silu',
            'nms_conf_thre': 0.05,
            'nms_iou_thre': 0.7,
            'legacy': False,
        }
        self.model.head = ZeroHead

        # Class name
        self.dataset.class_names = ['L']
