"""
Plot Precision-Recall curve for trained model
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import sys
import os

# Add damo-yolo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.base_models.core.ops import RepConv
from damo.utils.demo_utils import transform_img
from damo.structures.image_list import ImageList
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
from tqdm import tqdm


class Infer:
    def __init__(self, config_path, checkpoint_path, device='cuda', infer_size=[640, 640]):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.infer_size = infer_size

        # Load config
        self.config = parse_config(config_path)
        self.config.dataset.size_divisibility = 0

        # Build model
        self.model = build_local_model(self.config, self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'], strict=True)

        # Convert RepConv to deploy mode
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

        # Postprocess
        output = output[0].resize(origin_shape)
        bboxes = output.bbox
        scores = output.get_field('scores')
        cls_inds = output.get_field('labels')

        return bboxes, scores, cls_inds


def run_inference(infer, dataset_path, ann_file):
    """Run inference on dataset and return predictions"""

    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()

    results = []

    for img_id in tqdm(img_ids, desc="Running inference"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = Path(dataset_path) / img_info['file_name']

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue

        # Inference
        bboxes, scores, cls_inds = infer.forward(img)

        # Convert to COCO format
        bboxes = bboxes.cpu().numpy()
        scores = scores.cpu().numpy()
        cls_inds = cls_inds.cpu().numpy()

        for i in range(len(scores)):
            x1, y1, x2, y2 = bboxes[i]
            results.append({
                'image_id': img_id,
                'category_id': int(cls_inds[i]) + 1,  # Convert 0-indexed to 1-indexed for COCO
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'score': float(scores[i])
            })

    return results, coco_gt


def compute_pr_curve(coco_gt, results, iou_thresh=0.5):
    """Compute Precision-Recall curve"""

    if len(results) == 0:
        return np.array([0]), np.array([0]), 0

    # Create COCO results
    coco_dt = coco_gt.loadRes(results)

    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = [iou_thresh]
    coco_eval.evaluate()
    coco_eval.accumulate()

    # Extract precision and recall
    # precision has shape [T, R, K, A, M]
    # T: iou thresholds, R: recall thresholds, K: categories, A: areas, M: max dets
    precision = coco_eval.eval['precision'][0, :, 0, 0, 2]  # iou=0.5, all recall, cat 0, all area, maxDet=100
    recall = coco_eval.params.recThrs

    # Remove -1 values
    valid = precision >= 0
    precision = precision[valid]
    recall = recall[valid]

    # Compute AP
    ap = np.mean(precision) if len(precision) > 0 else 0

    return recall, precision, ap


def plot_pr_curves(pr_data, output_path='pr_curve.png', title='Precision-Recall Curve'):
    """Plot multiple PR curves"""

    plt.figure(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (name, (recall, precision, ap)) in enumerate(pr_data.items()):
        color = colors[i % len(colors)]
        plt.plot(recall, precision, color=color, linewidth=2,
                label=f'{name} (AP={ap:.1%})')
        plt.fill_between(recall, precision, alpha=0.1, color=color)

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved PR curve to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PR curve")
    parser.add_argument('--config', '-f', type=str, help='Config file path')
    parser.add_argument('--checkpoint', '-c', type=str, help='Checkpoint path')
    parser.add_argument('--dataset', '-d', type=str, help='Dataset images path')
    parser.add_argument('--ann_file', '-a', type=str, help='Annotation JSON file')
    parser.add_argument('--output', '-o', type=str, default='pr_curve.png', help='Output image path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--iou', type=float, nargs='+', default=[0.5], help='IoU threshold(s), e.g., --iou 0.5 0.75 0.9')
    parser.add_argument('--infer_size', type=int, nargs=2, default=[640, 640], help='Inference size')

    args = parser.parse_args()

    if args.config and args.checkpoint:
        print(f"Loading model from {args.checkpoint}")
        infer = Infer(args.config, args.checkpoint, args.device, args.infer_size)

        print(f"Running inference on {args.dataset}")
        results, coco_gt = run_inference(infer, args.dataset, args.ann_file)

        # Compute PR curves for multiple IoU thresholds
        pr_data = {}
        iou_list = args.iou

        for iou_thresh in iou_list:
            print(f"Computing PR curve (IoU={iou_thresh})")
            recall, precision, ap = compute_pr_curve(coco_gt, results, iou_thresh)
            pr_data[f'IoU={iou_thresh}'] = (recall, precision, ap)
            print(f"  AP@{iou_thresh}: {ap:.1%}")

        # Generate title
        if len(iou_list) == 1:
            title = f'Precision-Recall Curve (IoU={iou_list[0]})'
        else:
            title = 'Precision-Recall Curve (Multiple IoU)'

        plot_pr_curves(pr_data, args.output, title)
    else:
        print("Usage:")
        print("python plot_pr_curve.py -f saved_models/1class_S/config.py -c saved_models/1class_S/damoyolo_1class_S.pth -d datasets/dataset_S_coco/images -a datasets/dataset_S_coco/annotations/val.json -o pr_curve_S.png")
        print("")
        print("Multiple IoU thresholds:")
        print("python plot_pr_curve.py -f config.py -c model.pth -d images -a val.json -o pr_curve.png --iou 0.5 0.75 0.9")
