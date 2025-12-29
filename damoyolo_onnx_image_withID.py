#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DAMO-YOLO Image Detection
- Detects objects in a single image
- Assigns sequential numbers to detected objects
"""
import copy
import argparse

import cv2
import numpy as np

from damoyolo_onnx import DAMOYOLO


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image", type=str, required=True, help="Input image path")
    parser.add_argument("-m", "--model", type=str, required=True, help="ONNX model path")
    parser.add_argument("-c", "--classes", type=str, default=None, help="Classes file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output image path")

    # DAMO-YOLO parameters
    parser.add_argument('--score_th', type=float, default=0.8, help='Score threshold')
    parser.add_argument('--nms_th', type=float, default=0.8, help='NMS threshold')

    parser.add_argument("--cpu", action="store_true", help="Use CPU only")

    args = parser.parse_args()
    return args


def get_id_color(index):
    """Color based on ID"""
    temp_index = abs(int(index)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def main():
    args = get_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Cannot open image {args.image}")
        return

    # Load class names
    if args.classes:
        with open(args.classes, 'rt') as f:
            class_names = f.read().rstrip('\n').split('\n')
    else:
        class_names = None

    # Load model
    if args.cpu:
        providers = ['CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    model = DAMOYOLO(args.model, providers=providers)

    # Detection
    bboxes, scores, class_ids = model(image, score_th=args.score_th, nms_th=args.nms_th)

    debug_image = copy.deepcopy(image)

    # Count per class
    class_counts = {}
    for class_id in class_ids:
        class_id = int(class_id)
        class_counts[class_id] = class_counts.get(class_id, 0) + 1

    # Draw results with sequential numbers
    for idx, (bbox, score, class_id) in enumerate(zip(bboxes, scores, class_ids)):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        class_id = int(class_id)

        if args.score_th > score:
            continue

        color = get_id_color(idx)

        # Bounding box
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, thickness=2)

        # Label: class name + number
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = str(class_id)

        text = f'{class_name}:{idx + 1}'

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Background rectangle (filled with color)
        cv2.rectangle(debug_image, (x1, y1), (x1 + text_w + 4, y1 + text_h + 6), color, -1)

        # Text (white)
        cv2.putText(debug_image, text, (x1 + 2, y1 + text_h + 2),
                    font, font_scale, (255, 255, 255), thickness)

    # Info with black background
    info_font_scale = 0.6
    info_thickness = 1
    total_count = len(bboxes)
    info_text = f'N:{total_count}'

    # Add per-class counts
    if class_names:
        class_count_parts = []
        for cid, cnt in sorted(class_counts.items()):
            if cid < len(class_names):
                class_count_parts.append(f'{class_names[cid]}:{cnt}')
        if class_count_parts:
            info_text += ' (' + ', '.join(class_count_parts) + ')'

    (info_w, info_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, info_font_scale, info_thickness)
    cv2.rectangle(debug_image, (2, 2), (info_w + 8, info_h + 8), (0, 0, 0), -1)
    cv2.putText(debug_image, info_text, (5, info_h + 5),
                cv2.FONT_HERSHEY_SIMPLEX, info_font_scale, (255, 255, 255), thickness=info_thickness)

    # Output
    if args.output:
        cv2.imwrite(args.output, debug_image)
        print(f'Saved to {args.output}')

    print(f'Detected: {total_count} objects')
    for cid, cnt in sorted(class_counts.items()):
        if class_names and cid < len(class_names):
            print(f'  {class_names[cid]}: {cnt}')
        else:
            print(f'  Class {cid}: {cnt}')

    # Display
    cv2.imshow('DAMO-YOLO Detection', debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
