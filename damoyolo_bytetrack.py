#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DAMO-YOLO + ByteTrack
"""
import copy
import time
import argparse
import sys

import cv2
import numpy as np

sys.path.insert(0, 'bytetrack_sample')
from damoyolo_onnx import DAMOYOLO
from byte_tracker.byte_tracker import ByteTracker


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--video", type=str, default=None, help="Video file or camera index")
    parser.add_argument("-m", "--model", type=str, required=True, help="ONNX model path")
    parser.add_argument("-c", "--classes", type=str, default=None, help="Classes file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output video path")

    # DAMO-YOLO parameters
    parser.add_argument('--score_th', type=float, default=0.8, help='Score threshold')
    parser.add_argument('--nms_th', type=float, default=0.8, help='NMS threshold')

    # ByteTrack parameters
    parser.add_argument("--track_thresh", type=float, default=0.5)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--min_box_area", type=int, default=10)

    parser.add_argument("--cpu", action="store_true", help="Use CPU only")

    args = parser.parse_args()
    return args


class dict_dot_notation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def get_id_color(index):
    temp_index = abs(int(index)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def main():
    args = get_args()

    # Video input
    if args.video is None:
        cap = cv2.VideoCapture(0)
    elif args.video.isdigit():
        cap = cv2.VideoCapture(int(args.video))
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return

    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    if cap_fps == 0:
        cap_fps = 30

    # Output video
    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, cap_fps, (width, height))

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

    # Get input shape from model
    input_shape = tuple(model.input_shape)

    # Initialize tracker
    tracker = ByteTracker(
        args=dict_dot_notation({
            'track_thresh': args.track_thresh,
            'track_buffer': args.track_buffer,
            'match_thresh': args.match_thresh,
            'min_box_area': args.min_box_area,
            'mot20': False,
        }),
        frame_rate=int(cap_fps + 0.49),
    )

    track_id_dict = {}
    frame_count = 0

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        debug_image = copy.deepcopy(frame)

        # Detection
        bboxes, scores, class_ids = model(frame, score_th=args.score_th, nms_th=args.nms_th)

        # Prepare detections for ByteTrack: [x1, y1, x2, y2, score, class_id]
        if len(bboxes) > 0:
            detections = [[*b, s, l] for b, s, l in zip(bboxes, scores, class_ids)]
            detections = np.array(detections)
        else:
            detections = np.array([])

        # Tracking
        if len(detections) > 0:
            track_bboxes, track_scores, tracker_ids = tracker(
                detections,
                frame,
                input_shape,
            )
        else:
            track_bboxes, track_scores, tracker_ids = [], [], []

        # Assign sequential IDs
        for tracker_id in tracker_ids:
            if tracker_id not in track_id_dict:
                track_id_dict[tracker_id] = len(track_id_dict)

        elapsed_time = time.time() - start_time

        # Draw results
        for bbox, score, tracker_id in zip(track_bboxes, track_scores, tracker_ids):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            if args.score_th > score:
                continue

            color = get_id_color(track_id_dict[tracker_id])

            # Bounding box
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, thickness=2)

            # Label with background (inside box, top-left)
            text = f'{track_id_dict[tracker_id]}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # Background rectangle (filled with color)
            cv2.rectangle(debug_image, (x1, y1), (x1 + text_w + 4, y1 + text_h + 6), color, -1)

            # Text (white)
            cv2.putText(debug_image, text, (x1 + 2, y1 + text_h + 2),
                        font, font_scale, (255, 255, 255), thickness)

        # FPS with black background
        fps_text = f'FPS:{1/elapsed_time:.1f}'
        fps_font_scale = 0.6
        fps_thickness = 1
        (fps_w, fps_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, fps_thickness)
        cv2.rectangle(debug_image, (2, 2), (fps_w + 8, fps_h + 8), (0, 0, 0), -1)
        cv2.putText(debug_image, fps_text, (5, fps_h + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, (255, 255, 255), thickness=fps_thickness)

        # Object count with black background
        count_text = f'N:{len(track_bboxes)}'
        (count_w, count_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, fps_thickness)
        count_y = fps_h + 12
        cv2.rectangle(debug_image, (2, count_y), (count_w + 8, count_y + count_h + 6), (0, 0, 0), -1)
        cv2.putText(debug_image, count_text, (5, count_y + count_h + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, (255, 255, 255), thickness=fps_thickness)

        # Output
        if writer:
            writer.write(debug_image)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f'Frame {frame_count}, Objects: {len(track_bboxes)}, FPS: {1/elapsed_time:.1f}')

        cv2.imshow('DAMO-YOLO ByteTrack', debug_image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f'Done. Total frames: {frame_count}, Total tracked IDs: {len(track_id_dict)}')


if __name__ == '__main__':
    main()
