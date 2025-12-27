"""
Plot IoU distribution between predictions and ground truth
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.base_models.core.ops import RepConv
from damo.utils.demo_utils import transform_img
from damo.structures.image_list import ImageList
from pycocotools.coco import COCO
import cv2
from tqdm import tqdm


class Infer:
    def __init__(self, config_path, checkpoint_path, device='cuda', infer_size=[640, 640]):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.infer_size = infer_size

        self.config = parse_config(config_path)
        self.config.dataset.size_divisibility = 0

        self.model = build_local_model(self.config, self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'], strict=True)

        for layer in self.model.modules():
            if isinstance(layer, RepConv):
                layer.switch_to_deploy()

        self.model.eval()

    def _pad_image(self, img, target_size):
        n, c, h, w = img.shape
        assert n == 1
        target_size = [n, c, target_size[0], target_size[1]]
        pad_imgs = torch.zeros(*target_size)
        pad_imgs[:, :c, :h, :w].copy_(img)
        img_sizes = [img.shape[-2:]]
        pad_sizes = [pad_imgs.shape[-2:]]
        return ImageList(pad_imgs, img_sizes, pad_sizes)

    def preprocess(self, origin_img):
        img = transform_img(origin_img, 0,
                           **self.config.test.augment.transform,
                           infer_size=self.infer_size)
        oh, ow, _ = origin_img.shape
        img = self._pad_image(img.tensors, self.infer_size)
        img = img.to(self.device)
        return img, (ow, oh)

    def forward(self, origin_image):
        image, origin_shape = self.preprocess(origin_image)

        with torch.no_grad():
            output = self.model(image)

        output = output[0].resize(origin_shape)
        bboxes = output.bbox
        scores = output.get_field('scores')
        cls_inds = output.get_field('labels')

        return bboxes, scores, cls_inds


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


def get_iou_distribution(infer, dataset_path, ann_file, conf_thresh=0.5):
    """Get IoU distribution between predictions and GT"""

    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()

    all_ious = []
    matched_count = 0
    unmatched_pred = 0
    unmatched_gt = 0

    for img_id in tqdm(img_ids, desc="Computing IoU"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = Path(dataset_path) / img_info['file_name']

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Get predictions
        bboxes, scores, cls_inds = infer.forward(img)
        bboxes = bboxes.cpu().numpy()
        scores = scores.cpu().numpy()
        cls_inds = cls_inds.cpu().numpy()

        # Filter by confidence
        mask = scores >= conf_thresh
        pred_boxes = bboxes[mask]  # [x1, y1, x2, y2]
        pred_classes = cls_inds[mask]

        # Get GT annotations
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)

        gt_boxes = []
        gt_classes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_classes.append(ann['category_id'] - 1)  # 0-indexed

        # Match predictions to GT
        gt_matched = [False] * len(gt_boxes)

        for pred_box, pred_cls in zip(pred_boxes, pred_classes):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                if gt_matched[gt_idx]:
                    continue
                if pred_cls != gt_cls:
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

    return all_ious, matched_count, unmatched_pred, unmatched_gt


def plot_iou_distribution(ious, output_path, title='IoU Distribution'):
    """Plot IoU histogram"""

    plt.figure(figsize=(10, 6))

    bins = np.arange(0, 1.05, 0.05)
    plt.hist(ious, bins=bins, edgecolor='black', alpha=0.7)

    # Add vertical lines for common thresholds
    plt.axvline(x=0.5, color='r', linestyle='--', label='IoU=0.5')
    plt.axvline(x=0.75, color='orange', linestyle='--', label='IoU=0.75')
    plt.axvline(x=0.9, color='green', linestyle='--', label='IoU=0.9')

    mean_iou = np.mean(ious) if ious else 0
    plt.axvline(x=mean_iou, color='blue', linestyle='-', linewidth=2, label=f'Mean={mean_iou:.3f}')

    plt.xlabel('IoU', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])

    # Add statistics text
    stats_text = f'N={len(ious)}\nMean={mean_iou:.3f}\nMedian={np.median(ious):.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved IoU distribution to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot IoU distribution")
    parser.add_argument('--config', '-f', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset images path')
    parser.add_argument('--ann_file', '-a', type=str, required=True, help='Annotation JSON file')
    parser.add_argument('--output', '-o', type=str, default='iou_distribution.png', help='Output image path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--title', '-t', type=str, default='IoU Distribution', help='Plot title')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--infer_size', type=int, nargs=2, default=[640, 640], help='Inference size')

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    infer = Infer(args.config, args.checkpoint, args.device, args.infer_size)

    print(f"Computing IoU distribution on {args.dataset}")
    ious, matched, unmatched_pred, unmatched_gt = get_iou_distribution(
        infer, args.dataset, args.ann_file, args.conf
    )

    print(f"\nResults:")
    print(f"  Matched predictions: {matched}")
    print(f"  Unmatched predictions (FP): {unmatched_pred}")
    print(f"  Unmatched GT (FN): {unmatched_gt}")

    if ious:
        print(f"  Mean IoU: {np.mean(ious):.3f}")
        print(f"  Median IoU: {np.median(ious):.3f}")
        print(f"  IoU >= 0.5: {sum(1 for i in ious if i >= 0.5)} ({100*sum(1 for i in ious if i >= 0.5)/len(ious):.1f}%)")
        print(f"  IoU >= 0.75: {sum(1 for i in ious if i >= 0.75)} ({100*sum(1 for i in ious if i >= 0.75)/len(ious):.1f}%)")
        print(f"  IoU >= 0.9: {sum(1 for i in ious if i >= 0.9)} ({100*sum(1 for i in ious if i >= 0.9)/len(ious):.1f}%)")

    plot_iou_distribution(ious, args.output, args.title)
