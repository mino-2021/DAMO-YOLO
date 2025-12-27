#!/usr/bin/env python
"""Visualize predictions with IoU < threshold"""
import argparse
import os
import numpy as np
import torch
import cv2
from pathlib import Path

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29504'

import sys
sys.path.insert(0, '.')

from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.base_models.core.ops import RepConv
from damo.dataset import build_dataset
from damo.dataset.build import build_dataloader


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area


def draw_boxes(img, pred_box, gt_box, iou, pred_score):
    """Draw prediction (red) and GT (green) boxes"""
    img = img.copy()

    # GT box - green
    x1, y1, x2, y2 = [int(v) for v in gt_box]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(img, 'GT', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Pred box - red
    x1, y1, x2, y2 = [int(v) for v in pred_box]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(img, f'Pred (IoU={iou:.3f}, score={pred_score:.2f})', (x1, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config', required=True, help='Config file')
    parser.add_argument('-c', '--ckpt', required=True, help='Checkpoint')
    parser.add_argument('-o', '--output_dir', default='low_iou_images', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.95, help='IoU threshold (show below this)')
    parser.add_argument('--max_images', type=int, default=20, help='Max images to save')
    args = parser.parse_args()

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend=backend, init_method='env://')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = parse_config(args.config)

    # Build model
    model = build_local_model(config, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)

    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()
    model.eval()

    # Build dataset
    val_dataset = build_dataset(config, config.dataset.val_ann, is_train=False)
    val_loader = build_dataloader(val_dataset, config.test.augment,
                                  batch_size=1, num_workers=0, is_train=False, size_div=32)

    coco_gt = val_dataset[0].coco
    coco_image_ids = coco_gt.getImgIds()

    # Get image directory from dataset
    img_dir = val_dataset[0].root

    os.makedirs(args.output_dir, exist_ok=True)

    cpu_device = torch.device('cpu')
    idx = 0
    saved_count = 0
    all_iou_cases = []

    print(f"Finding predictions...")

    for _, batch in enumerate(val_loader[0]):
        images, targets, image_ids = batch

        with torch.no_grad():
            output = model(images.to(device))
            output = [o.to(cpu_device) if o is not None else o for o in output]

        for pred in output:
            if pred is None:
                idx += 1
                continue

            img_id = coco_image_ids[idx]
            img_info = coco_gt.loadImgs(img_id)[0]
            orig_w, orig_h = img_info['width'], img_info['height']
            pred = pred.resize((orig_w, orig_h))

            # Get predictions
            boxes = pred.bbox.cpu().numpy()
            scores = pred.get_field('scores').cpu().numpy()
            labels = pred.get_field('labels').cpu().numpy()

            # Filter by confidence
            mask = scores >= args.conf
            pred_boxes = boxes[mask]
            pred_scores = scores[mask]
            pred_labels = labels[mask]

            # Get GT
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            anns = coco_gt.loadAnns(ann_ids)

            gt_boxes = []
            gt_labels = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                gt_boxes.append([x, y, x + w, y + h])
                gt_labels.append(ann['category_id'] - 1)

            # Match and find low IoU cases
            gt_matched = [False] * len(gt_boxes)

            for pred_box, pred_score, pred_label in zip(pred_boxes, pred_scores, pred_labels):
                best_iou = 0
                best_gt_idx = -1
                best_gt_box = None

                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_matched[gt_idx]:
                        continue
                    if int(pred_label) != int(gt_label):
                        continue

                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        best_gt_box = gt_box

                if best_gt_idx >= 0 and best_iou > 0:
                    gt_matched[best_gt_idx] = True
                    all_iou_cases.append({
                        'img_info': img_info,
                        'pred_box': pred_box,
                        'gt_box': best_gt_box,
                        'iou': best_iou,
                        'score': pred_score,
                        'label': pred_label
                    })

            idx += 1

    # Sort by IoU (lowest first)
    all_iou_cases.sort(key=lambda x: x['iou'])

    # Filter low IoU cases
    low_iou_cases = [c for c in all_iou_cases if c['iou'] < args.iou_thresh]

    print(f"\nFound {len(low_iou_cases)} predictions with IoU < {args.iou_thresh}")
    print(f"Saving top {min(args.max_images, len(low_iou_cases))} worst cases...")

    for i, case in enumerate(low_iou_cases[:args.max_images]):
        img_info = case['img_info']
        img_path = os.path.join(img_dir, img_info['file_name'])

        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot load {img_path}")
            continue

        img_vis = draw_boxes(img, case['pred_box'], case['gt_box'], case['iou'], case['score'])

        # Save
        out_name = f"iou_{case['iou']:.3f}_{Path(img_info['file_name']).stem}.jpg"
        out_path = os.path.join(args.output_dir, out_name)

        # Resize for viewing if too large
        h, w = img_vis.shape[:2]
        if max(h, w) > 1500:
            scale = 1500 / max(h, w)
            img_vis = cv2.resize(img_vis, (int(w * scale), int(h * scale)))

        cv2.imwrite(out_path, img_vis)
        print(f"  [{i+1}] IoU={case['iou']:.3f}, score={case['score']:.2f} -> {out_name}")

    # Save best 10 IoU cases
    print(f"\nSaving top 10 best IoU cases...")
    best_iou_cases = sorted(all_iou_cases, key=lambda x: x['iou'], reverse=True)[:10]

    for i, case in enumerate(best_iou_cases):
        img_info = case['img_info']
        img_path = os.path.join(img_dir, img_info['file_name'])

        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot load {img_path}")
            continue

        img_vis = draw_boxes(img, case['pred_box'], case['gt_box'], case['iou'], case['score'])

        out_name = f"best_{i+1}_iou_{case['iou']:.3f}_{Path(img_info['file_name']).stem}.jpg"
        out_path = os.path.join(args.output_dir, out_name)

        h, w = img_vis.shape[:2]
        if max(h, w) > 1500:
            scale = 1500 / max(h, w)
            img_vis = cv2.resize(img_vis, (int(w * scale), int(h * scale)))

        cv2.imwrite(out_path, img_vis)
        print(f"  [Best {i+1}] IoU={case['iou']:.3f}, score={case['score']:.2f} -> {out_name}")

    print(f"\nSaved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
