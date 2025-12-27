#!/usr/bin/env python
"""Plot IoU distribution by class in one graph with AP95"""
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config', required=True, help='Config file')
    parser.add_argument('-c', '--ckpt', required=True, help='Checkpoint')
    parser.add_argument('-o', '--output', default='iou_dist_by_class.png', help='Output path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('-t', '--title', default='IoU Distribution by Class', help='Plot title')
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

    # Get class names
    class_names = config.dataset.class_names
    num_classes = len(class_names)

    # IoU per class
    ious_by_class = {i: [] for i in range(num_classes)}

    # For AP calculation - store all predictions
    all_results = []

    cpu_device = torch.device('cpu')
    idx = 0

    for _, batch in enumerate(tqdm(val_loader[0], desc="Computing IoU")):
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

            # Store all predictions for AP calculation (no filtering)
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                all_results.append({
                    'image_id': img_id,
                    'category_id': int(label) + 1,  # 1-indexed for COCO
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'score': float(score)
                })

            # Filter by confidence for IoU calculation
            mask = scores >= args.conf
            pred_boxes = boxes[mask]
            pred_labels = labels[mask]

            # Get GT
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            anns = coco_gt.loadAnns(ann_ids)

            gt_boxes = []
            gt_labels = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                gt_boxes.append([x, y, x + w, y + h])
                gt_labels.append(ann['category_id'] - 1)  # 0-indexed

            # Match predictions to GT
            gt_matched = [False] * len(gt_boxes)

            for pred_box, pred_label in zip(pred_boxes, pred_labels):
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_matched[gt_idx]:
                        continue
                    if int(pred_label) != int(gt_label):
                        continue

                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0 and best_iou > 0:
                    class_idx = int(pred_label)
                    ious_by_class[class_idx].append(best_iou)
                    gt_matched[best_gt_idx] = True

            idx += 1

    # Calculate AP95 per class using COCO API
    ap95_by_class = {}
    if all_results:
        coco_dt = coco_gt.loadRes(all_results)
        for class_idx in range(num_classes):
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.params.iouThrs = np.array([0.95])
            coco_eval.params.catIds = [class_idx + 1]  # 1-indexed
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            ap95_by_class[class_idx] = coco_eval.stats[0] * 100  # Convert to percentage

    # Print results by class
    print(f"\n=== Results by Class ===")
    for class_idx in range(num_classes):
        ious = ious_by_class[class_idx]
        class_name = class_names[class_idx]
        ap95 = ap95_by_class.get(class_idx, 0)
        if ious:
            print(f"\n{class_name} (class {class_idx}):")
            print(f"  N: {len(ious)}")
            print(f"  Mean IoU: {np.mean(ious):.3f}")
            print(f"  Median IoU: {np.median(ious):.3f}")
            print(f"  IoU >= 0.9: {sum(1 for i in ious if i >= 0.9)} ({100*sum(1 for i in ious if i >= 0.9)/len(ious):.1f}%)")
            print(f"  IoU >= 0.95: {sum(1 for i in ious if i >= 0.95)} ({100*sum(1 for i in ious if i >= 0.95)/len(ious):.1f}%)")
            print(f"  AP95: {ap95:.1f}%")
        else:
            print(f"\n{class_name} (class {class_idx}): No predictions")

    # Plot - 縦に2つのグラフ
    fig, axes = plt.subplots(num_classes, 1, figsize=(12, 4 * num_classes))
    if num_classes == 1:
        axes = [axes]

    bins = np.arange(0, 1.05, 0.05)
    colors = ['steelblue', 'coral', 'green', 'purple', 'orange']

    # Histogram for each class (separate subplot)
    for class_idx in range(num_classes):
        ax = axes[class_idx]
        ious = ious_by_class[class_idx]
        class_name = class_names[class_idx]
        ap95 = ap95_by_class.get(class_idx, 0)

        if ious:
            mean_iou = np.mean(ious)
            iou95_rate = 100 * sum(1 for i in ious if i >= 0.95) / len(ious)

            ax.hist(ious, bins=bins, alpha=0.7,
                    color=colors[class_idx % len(colors)], edgecolor='black')

            # Reference lines
            ax.axvline(x=0.9, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvline(x=0.95, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvline(x=mean_iou, color='blue', linestyle='-', linewidth=2, alpha=0.8)

            # Title with stats
            title = f'{class_name}: N={len(ious)}, Mean={mean_iou:.3f}, IoU≥0.95={iou95_rate:.1f}%, AP95={ap95:.1f}%'
            ax.set_title(title, fontsize=12, fontweight='bold')

            # Stats text box
            stats_text = f'Mean={mean_iou:.3f}\nMedian={np.median(ious):.3f}\nIoU≥0.95: {sum(1 for i in ious if i >= 0.95)}/{len(ious)}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.set_title(f'{class_name}: No predictions', fontsize=12)

        ax.set_xlabel('IoU', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])

    fig.suptitle(args.title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved to: {args.output}")
    plt.close()


if __name__ == '__main__':
    main()
