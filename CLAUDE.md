# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DAMO-YOLO is a fast and accurate object detection framework developed by Alibaba DAMO Data Analytics and Intelligence Lab. It extends YOLO with Neural Architecture Search (NAS) backbones, Reparameterized Generalized-FPN (RepGFPN), lightweight ZeroHead with AlignedOTA label assignment, and distillation enhancement.

## Common Commands

### Training

```bash
# Multi-GPU distributed training (Linux)
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/damoyolo_tinynasL25_S.py

# Single GPU training (Windows compatible)
python train_single_gpu.py -f configs/damoyolo_tinynasL20_T_2class.py

# Training with knowledge distillation
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/damoyolo_tinynasL20_T.py --tea_config configs/damoyolo_tinynasL25_S.py --tea_ckpt path/to/teacher.pth
```

### Evaluation

```bash
python -m torch.distributed.launch --nproc_per_node=8 tools/eval.py -f configs/damoyolo_tinynasL25_S.py --ckpt /path/to/checkpoint.pth
```

### Inference Demo

```bash
# Image inference with torch model
python tools/demo.py image -f ./configs/damoyolo_tinynasL25_S.py --engine ./model.pth --conf 0.6 --infer_size 640 640 --device cuda --path ./image.jpg

# Video inference
python tools/demo.py video -f ./configs/damoyolo_tinynasL25_S.py --engine ./model.onnx --conf 0.6 --infer_size 640 640 --device cuda --path video.mp4

# Camera inference
python tools/demo.py camera -f ./configs/damoyolo_tinynasL25_S.py --engine ./model.trt --conf 0.6 --infer_size 640 640 --device cuda --camid 0
```

### Model Export

```bash
# Export to ONNX
python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c model.pth --batch_size 1 --img_size 640

# Export to TensorRT with end-to-end NMS
python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c model.pth --batch_size 1 --img_size 640 --trt --end2end --trt_type fp16
```

## Architecture

### Directory Structure

- `configs/` - Model configuration files (backbone, neck, head, training params)
- `damo/` - Core library
  - `base_models/` - Backbones (TinyNAS), necks (GiraffeFPN), heads (ZeroHead), losses
  - `apis/` - Training and inference APIs
  - `dataset/` - Data loading, augmentation, COCO evaluation
  - `config/` - Configuration system and dataset path catalog
- `tools/` - Training, evaluation, demo, and conversion scripts
- `datasets/` - Symlinked dataset directory (COCO format expected)

### Configuration System

Configs inherit from `damo.config.Config` and define:
- **Model**: backbone structure, neck channels, head (num_classes, NMS thresholds)
- **Training**: epochs, batch_size, learning rate, augmentation, finetune_path
- **Dataset**: train_ann, val_ann (references to `damo/config/paths_catalog.py`)

### Custom Dataset Setup

1. Convert dataset to COCO format (images + annotations JSON)
2. Add dataset paths to `damo/config/paths_catalog.py`:
   ```python
   'custom_train_coco': {
       'img_dir': 'custom_dataset/images',
       'ann_file': 'custom_dataset/annotations/train.json'
   }
   ```
3. Create config file with correct `num_classes` and `class_names`
4. Set `self.train.finetune_path` for transfer learning from pretrained weights

### Key Classes

- `ZeroHead` - Detection head with configurable `num_classes`, `nms_conf_thre`, `nms_iou_thre`
- `RepConv` - Reparameterizable convolution (switches to deploy mode for inference)
- `Trainer` (damo.apis) - Handles training loop, checkpointing, evaluation
- `Infer` (tools/demo.py) - Inference engine supporting torch/onnx/tensorRT

## Platform Notes

- Windows: Use `train_single_gpu.py` with gloo backend (nccl not supported)
- Linux: Use `torch.distributed.launch` with nccl backend for multi-GPU
- PYTHONPATH must include project root: `export PYTHONPATH=$PWD:$PYTHONPATH`

## Analysis Scripts

### IoU Distribution by Class

クラス別のIoU分布をヒストグラムで可視化し、AP95を計算する。

```bash
python plot_iou_dist_by_class.py \
  -f configs/damoyolo_tinynasL20_T_2class_norot_aug.py \
  -c workdirs/damoyolo_tinynasL20_T_2class_norot_aug/epoch_200_ckpt.pth \
  -o iou_dist_by_class.png \
  -t "Model Name - IoU Distribution by Class" \
  --conf 0.5
```

**出力:**
- クラス別のIoUヒストグラム（縦に並べて表示）
- 各クラスのMean IoU、Median IoU、IoU≥0.95率、AP95
- 緑線: IoU=0.9、赤線: IoU=0.95、青線: Mean IoU

### Best/Worst IoU Visualization

IoUのベスト・ワーストケースを画像で可視化する。

```bash
python visualize_low_iou.py \
  -f configs/damoyolo_tinynasL20_T_2class_norot_aug.py \
  -c workdirs/damoyolo_tinynasL20_T_2class_norot_aug/epoch_200_ckpt.pth \
  -o output_dir/iou_images \
  --iou_thresh 0.95 \
  --max_images 20 \
  --conf 0.5
```

**出力:**
- `iou_X.XXX_*.jpg` - IoU < threshold のワーストケース
- `best_N_iou_X.XXX_*.jpg` - IoU上位10件のベストケース
- 緑枠: Ground Truth、赤枠: 予測

### Training Loss Graph

訓練ログからロスグラフを生成する。

```bash
python plot_training_loss.py \
  -l workdirs/damoyolo_tinynasL20_T_2class_norot_aug/2025-12-23-0802 \
  -o training_loss.png
```

### PR Curve

Precision-Recall曲線を描画する。

```bash
python plot_pr_curve.py \
  -f configs/damoyolo_tinynasL20_T_2class_norot_aug.py \
  -c workdirs/damoyolo_tinynasL20_T_2class_norot_aug/epoch_200_ckpt.pth \
  -o pr_curve.png
```

## Dataset Preparation Scripts

### 2-Class Dataset (S and L) - Aligned Split

L_norotと同じ検証画像を使用する2クラスデータセットを作成する。

```bash
python prepare_dataset_2class_norot_aligned.py
```

**設定（スクリプト内）:**
- `dataset_L_path`: Lクラス画像のパス
- `dataset_S_path`: Sクラス画像のパス
- `csv_L_file`: Lクラスのアノテーションcsv
- `csv_S_file`: Sクラスのアノテーションcsv
- `output_path`: 出力先
- `L_VAL_IMAGES`: 検証に使用するL画像のリスト（L_norotと一致させる）

## Saved Models

訓練済みモデルは `saved_models/` に保存:

```
saved_models/
├── 1class_L/              # Lクラス単独（回転あり）
├── 1class_L_norot/        # Lクラス単独（回転なし）
├── 1class_S/              # Sクラス単独（回転あり）
├── 1class_S_norot/        # Sクラス単独（回転なし）
├── 2class_SL/             # 2クラス（回転あり）
├── 2class_SL_norot/       # 2クラス（回転なし）
├── 2class_SL_norot_aug/   # 2クラス（回転なし、旧データ分割）
└── 2class_SL_norot_aug_aligned/  # 2クラス（回転なし、整合データ分割）
    ├── config.py
    ├── damoyolo_2class_norot_aug_aligned.pth
    ├── iou_dist_2class_norot_aug_aligned.png
    ├── prepare_dataset_2class_norot_aligned.py
    ├── training_log.txt
    └── iou_images/
        ├── iou_*.jpg (worst cases)
        └── best_*.jpg (best cases)
```
