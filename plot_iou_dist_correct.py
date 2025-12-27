#!/usr/bin/env python
"""Plot IoU distribution using DAMO dataloader (correct preprocessing)"""
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29503'

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
    parser.add_argument('-o', '--output', default='iou_distribution.png', help='Output path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('-t', '--title', default='IoU Distribution', help='Plot title')
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

    # Run inference and compute IoU
    all_ious = []
    matched_count = 0
    unmatched_pred = 0
    unmatched_gt = 0

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

            # Filter by confidence
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
                    all_ious.append(best_iou)
                    gt_matched[best_gt_idx] = True
                    matched_count += 1
                else:
                    unmatched_pred += 1

            unmatched_gt += sum(1 for m in gt_matched if not m)
            idx += 1

    # Print results
    print(f"\nResults:")
    print(f"  Matched predictions: {matched_count}")
    print(f"  Unmatched predictions (FP): {unmatched_pred}")
    print(f"  Unmatched GT (FN): {unmatched_gt}")

    if all_ious:
        print(f"  Mean IoU: {np.mean(all_ious):.3f}")
        print(f"  Median IoU: {np.median(all_ious):.3f}")
        print(f"  IoU >= 0.5: {sum(1 for i in all_ious if i >= 0.5)} ({100*sum(1 for i in all_ious if i >= 0.5)/len(all_ious):.1f}%)")
        print(f"  IoU >= 0.75: {sum(1 for i in all_ious if i >= 0.75)} ({100*sum(1 for i in all_ious if i >= 0.75)/len(all_ious):.1f}%)")
        print(f"  IoU >= 0.9: {sum(1 for i in all_ious if i >= 0.9)} ({100*sum(1 for i in all_ious if i >= 0.9)/len(all_ious):.1f}%)")
        print(f"  IoU >= 0.95: {sum(1 for i in all_ious if i >= 0.95)} ({100*sum(1 for i in all_ious if i >= 0.95)/len(all_ious):.1f}%)")

    # Plot
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 1.05, 0.05)
    plt.hist(all_ious, bins=bins, edgecolor='black', alpha=0.7)

    plt.axvline(x=0.5, color='r', linestyle='--', label='IoU=0.5')
    plt.axvline(x=0.75, color='orange', linestyle='--', label='IoU=0.75')
    plt.axvline(x=0.9, color='green', linestyle='--', label='IoU=0.9')
    plt.axvline(x=0.95, color='purple', linestyle='--', label='IoU=0.95')

    mean_iou = np.mean(all_ious) if all_ious else 0
    plt.axvline(x=mean_iou, color='blue', linestyle='-', linewidth=2, label=f'Mean={mean_iou:.3f}')

    plt.xlabel('IoU', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(args.title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])

    stats_text = f'N={len(all_ious)}\nMean={mean_iou:.3f}\nMedian={np.median(all_ious) if all_ious else 0:.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved to: {args.output}")
    plt.close()


if __name__ == '__main__':
    main()
