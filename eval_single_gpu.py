#!/usr/bin/env python
# Single GPU evaluation script for DAMO-YOLO
import argparse
import os
import numpy as np
import torch
from loguru import logger

from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.base_models.core.ops import RepConv
from damo.dataset import build_dataset
from damo.dataset.build import build_dataloader
from damo.apis.detector_inference import compute_on_dataset
from damo.dataset.datasets.evaluation import evaluate
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


def make_parser():
    parser = argparse.ArgumentParser('Damo-Yolo single GPU eval parser')
    parser.add_argument('-f', '--config_file', required=True, type=str, help='config file path')
    parser.add_argument('-c', '--ckpt', required=True, type=str, help='checkpoint path')
    parser.add_argument('--iou', type=float, nargs='+', default=[0.5, 0.75, 0.9], help='IoU thresholds')
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    # Setup single GPU environment
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'

    torch.cuda.set_device(0)
    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method='env://')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = parse_config(args.config_file)

    # Build model
    model = build_local_model(config, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)

    # Switch RepConv to deploy mode
    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()

    model.eval()
    model.cuda()

    # Build dataset and dataloader
    val_dataset = build_dataset(config, config.dataset.val_ann, is_train=False)
    val_loader = build_dataloader(val_dataset, config.test.augment,
                                  batch_size=1, num_workers=0, is_train=False, size_div=32)

    # Run inference
    print(f"Running inference on {len(val_dataset[0])} images...")

    model.eval()
    predictions = {}
    cpu_device = torch.device('cpu')
    for _, batch in enumerate(tqdm(val_loader[0])):
        images, targets, image_ids = batch
        with torch.no_grad():
            output = model(images.to(device))
            output = [o.to(cpu_device) if o is not None else o for o in output]
        predictions.update({img_id: result for img_id, result in zip(image_ids, output)})

    # Convert predictions to list format
    pred_image_ids = list(sorted(predictions.keys()))
    predictions_list = [predictions[i] for i in pred_image_ids]

    # Prepare COCO format results
    coco_gt = val_dataset[0].coco
    coco_image_ids = coco_gt.getImgIds()
    results = []

    for idx, pred in enumerate(predictions_list):
        if pred is None:
            continue

        img_id = coco_image_ids[idx]
        img_info = coco_gt.loadImgs(img_id)[0]
        orig_w, orig_h = img_info['width'], img_info['height']
        pred = pred.resize((orig_w, orig_h))

        boxes = pred.bbox.cpu().numpy()
        scores = pred.get_field('scores').cpu().numpy()
        labels = pred.get_field('labels').cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            results.append({
                'image_id': int(img_id),
                'category_id': int(label) + 1,  # 1-indexed for COCO
                'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                'score': float(score)
            })

    print(f"Total predictions: {len(results)}")

    # Evaluate with COCO
    coco_dt = coco_gt.loadRes(results)

    # Standard evaluation
    print("\n=== Standard COCO Evaluation (IoU=0.50:0.95) ===")
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Custom IoU thresholds
    for iou in args.iou:
        print(f"\n=== AP@{iou} ===")
        coco_eval_iou = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_iou.params.iouThrs = np.array([iou])
        coco_eval_iou.evaluate()
        coco_eval_iou.accumulate()
        coco_eval_iou.summarize()
        ap = coco_eval_iou.stats[0]
        print(f"\n*** AP{int(iou*100)} = {ap*100:.1f}% ***")


if __name__ == '__main__':
    main()
