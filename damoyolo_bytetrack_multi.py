#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DAMO-YOLO + ByteTrack (Multi-Class)
- Tracks each class separately with independent trackers
- Track IDs are formatted as "classID_trackID" (e.g., "0_1", "1_2")
"""
import copy
import time
import argparse
import sys

import cv2
import numpy as np

sys.path.insert(0, 'bytetrack_sample')
from damoyolo_onnx import DAMOYOLO
from byte_tracker.tracker.byte_tracker import BYTETracker


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


class MultiClassByteTrack:
    """Multi-class ByteTrack - separate tracker per class"""

    def __init__(self, fps, track_thresh=0.5, track_buffer=30, match_thresh=0.8, min_box_area=10):
        self.fps = fps
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        self.tracker_dict = {}  # class_id -> BYTETracker

    def update(self, bboxes, scores, class_ids, img_info, img_size):
        """
        Update trackers for all classes
        Returns: (track_ids, bboxes, scores, class_ids)
        """
        result_ids = []
        result_bboxes = []
        result_scores = []
        result_class_ids = []

        # Get unique classes in this frame
        unique_classes = np.unique(class_ids)

        for class_id in unique_classes:
            class_id = int(class_id)

            # Create tracker for new class if needed
            if class_id not in self.tracker_dict:
                self.tracker_dict[class_id] = BYTETracker(
                    args=dict_dot_notation({
                        'track_thresh': self.track_thresh,
                        'track_buffer': self.track_buffer,
                        'match_thresh': self.match_thresh,
                        'mot20': False,
                    }),
                    frame_rate=self.fps,
                )

            # Filter detections for this class
            mask = class_ids == class_id
            class_bboxes = bboxes[mask]
            class_scores = scores[mask]

            if len(class_bboxes) == 0:
                continue

            # Prepare detections: [x1, y1, x2, y2, score]
            detections = np.column_stack([class_bboxes, class_scores])

            # Update tracker
            online_targets = self.tracker_dict[class_id].update(
                detections,
                img_info,
                img_size,
            )

            # Collect results
            for target in online_targets:
                tlwh = target.tlwh
                if tlwh[2] * tlwh[3] > self.min_box_area:
                    # Convert tlwh to tlbr (x1, y1, x2, y2)
                    bbox = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
                    track_id = f"{class_id}_{target.track_id}"

                    result_ids.append(track_id)
                    result_bboxes.append(bbox)
                    result_scores.append(target.score)
                    result_class_ids.append(class_id)

        return result_ids, result_bboxes, result_scores, result_class_ids


def get_class_color(class_id):
    """Fixed color per class"""
    colors = [
        (255, 100, 100),  # Class 0: Light blue
        (100, 255, 100),  # Class 1: Light green
        (100, 100, 255),  # Class 2: Light red
        (255, 255, 100),  # Class 3: Cyan
        (255, 100, 255),  # Class 4: Magenta
        (100, 255, 255),  # Class 5: Yellow
    ]
    return colors[class_id % len(colors)]


def get_id_color(index):
    """Color based on track ID"""
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

    # Initialize multi-class tracker
    tracker = MultiClassByteTrack(
        fps=int(cap_fps + 0.49),
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        min_box_area=args.min_box_area,
    )

    track_id_dict = {}  # For sequential ID assignment
    class_counts = {}   # Count per class
    frame_count = 0

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        debug_image = copy.deepcopy(frame)
        frame_height, frame_width = frame.shape[:2]

        # Detection
        bboxes, scores, class_ids = model(frame, score_th=args.score_th, nms_th=args.nms_th)

        # Tracking
        if len(bboxes) > 0:
            bboxes_np = np.array(bboxes)
            scores_np = np.array(scores)
            class_ids_np = np.array(class_ids)

            track_ids, track_bboxes, track_scores, track_class_ids = tracker.update(
                bboxes_np,
                scores_np,
                class_ids_np,
                [frame_height, frame_width],
                input_shape,
            )
        else:
            track_ids, track_bboxes, track_scores, track_class_ids = [], [], [], []

        # Assign sequential IDs per class
        for track_id in track_ids:
            if track_id not in track_id_dict:
                track_id_dict[track_id] = len(track_id_dict)

        elapsed_time = time.time() - start_time

        # Count per class
        class_counts = {}
        for class_id in track_class_ids:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        # Draw results
        for track_id, bbox, score, class_id in zip(track_ids, track_bboxes, track_scores, track_class_ids):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            if args.score_th > score:
                continue

            # Extract the per-class track ID from "classID_trackID" format
            per_class_id = track_id.split('_')[1] if '_' in track_id else track_id

            # Use track ID-based color (different color per object)
            track_num = int(per_class_id) if per_class_id.isdigit() else hash(track_id)
            color = get_id_color(track_num)

            # Bounding box
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, thickness=2)

            # Label: class name + track ID
            if class_names and class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = str(class_id)

            text = f'{class_name}:{per_class_id}'

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

        # Object count per class with black background
        count_y = fps_h + 12
        total_count = len(track_bboxes)
        count_text = f'N:{total_count}'

        # Add per-class counts
        if class_names:
            class_count_parts = []
            for cid, cnt in sorted(class_counts.items()):
                if cid < len(class_names):
                    class_count_parts.append(f'{class_names[cid]}:{cnt}')
            if class_count_parts:
                count_text += ' (' + ', '.join(class_count_parts) + ')'

        (count_w, count_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, fps_thickness)
        cv2.rectangle(debug_image, (2, count_y), (count_w + 8, count_y + count_h + 6), (0, 0, 0), -1)
        cv2.putText(debug_image, count_text, (5, count_y + count_h + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, (255, 255, 255), thickness=fps_thickness)

        # Output
        if writer:
            writer.write(debug_image)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f'Frame {frame_count}, Objects: {len(track_bboxes)}, FPS: {1/elapsed_time:.1f}')

        cv2.imshow('DAMO-YOLO ByteTrack Multi-Class', debug_image)
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
