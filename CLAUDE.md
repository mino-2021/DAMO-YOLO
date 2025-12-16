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
