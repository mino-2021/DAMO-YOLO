#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import copy

import cv2
import numpy as np
import onnxruntime


class DAMOYOLO(object):
    def __init__(
        self,
        model_path,
        max_num=500,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):

        # パラメータ
        self.max_num = max_num

        # モデル読み込み
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_detail = self.onnx_session.get_inputs()[0]
        self.input_name = self.input_detail.name

        # 各種設定
        self.input_shape = self.input_detail.shape[2:]

    def __call__(self, image, score_th=0.05, nms_th=0.8):
        temp_image = copy.deepcopy(image)
        image_height, image_width = image.shape[0], image.shape[1]

        # 前処理
        image, ratio = self._preprocess(temp_image, self.input_shape)

        # 推論実施
        results = self.onnx_session.run(
            None,
            {self.input_name: image[None, :, :, :]},
        )

        # 後処理
        scores = results[0]
        bboxes = results[1]
        bboxes, scores, class_ids = self._postprocess(
            scores,
            bboxes,
            score_th,
            nms_th,
        )

        decode_ratio = min(image_height / int(image_height * ratio),
                           image_width / int(image_width * ratio))
        if len(bboxes) > 0:
            bboxes = bboxes * decode_ratio

        return bboxes, scores, class_ids

    def _preprocess(self, image, input_size, swap=(2, 0, 1)):
        temp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(image.shape) == 3:
            padded_image = np.ones((input_size[0], input_size[1], 3),
                                   dtype=np.uint8)
        else:
            padded_image = np.ones(input_size, dtype=np.uint8)

        ratio = min(input_size[0] / temp_image.shape[0],
                    input_size[1] / temp_image.shape[1])
        resized_image = cv2.resize(
            temp_image,
            (int(temp_image.shape[1] * ratio), int(
                temp_image.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_image = resized_image.astype(np.uint8)

        padded_image[:int(temp_image.shape[0] *
                          ratio), :int(temp_image.shape[1] *
                                       ratio)] = resized_image
        padded_image = padded_image.transpose(swap)
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)

        return padded_image, ratio

    def _postprocess(
        self,
        scores,
        bboxes,
        score_th,
        nms_th,
    ):
        batch_size = bboxes.shape[0]
        for i in range(batch_size):
            if not bboxes[i].shape[0]:
                continue
            bboxes, scores, class_ids = self._multiclass_nms(
                bboxes[i],
                scores[i],
                score_th,
                nms_th,
                self.max_num,
            )

        return bboxes, scores, class_ids

    def _multiclass_nms(
        self,
        bboxes,
        scores,
        score_th,
        nms_th,
        max_num=100,
        score_factors=None,
    ):
        num_classes = scores.shape[1]
        if bboxes.shape[1] > 4:
            pass
        else:
            bboxes = np.broadcast_to(
                bboxes[:, None],
                (bboxes.shape[0], num_classes, 4),
            )
        valid_mask = scores > score_th
        bboxes = bboxes[valid_mask]

        if score_factors is not None:
            scores = scores * score_factors[:, None]
        scores = scores[valid_mask]

        np_labels = valid_mask.nonzero()[1]

        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(),
            scores.tolist(),
            score_th,
            nms_th,
        )

        if max_num > 0:
            indices = indices[:max_num]

        if len(indices) > 0:
            bboxes = bboxes[indices]
            scores = scores[indices]
            np_labels = np_labels[indices]
            return bboxes, scores, np_labels
        else:
            return np.array([]), np.array([]), np.array([])

    def draw(
        self,
        image,
        score_th,
        bboxes,
        scores,
        class_ids,
        class_names,
        thickness=3,
    ):
        debug_image = copy.deepcopy(image)

        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(
                bbox[3])

            if score_th > score:
                continue

            color = self._get_color(class_id)

            # バウンディングボックス
            debug_image = cv2.rectangle(
                debug_image,
                (x1, y1),
                (x2, y2),
                color,
                thickness=thickness,
            )

            # クラスID、スコア
            score_text = '%.2f' % score
            text = '%s:%s' % (str(class_names[int(class_id)]), score_text)
            debug_image = cv2.putText(
                debug_image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                thickness=thickness,
            )

        return debug_image

    def _get_color(self, index):
        temp_index = abs(int(index + 5)) * 3
        color = (
            (29 * temp_index) % 255,
            (17 * temp_index) % 255,
            (37 * temp_index) % 255,
        )
        return color


def main():
    parser = argparse.ArgumentParser(description='DAMO-YOLO ONNX Inference')
    parser.add_argument('-m', '--model', required=True, help='ONNX model path')
    parser.add_argument('-v', '--video', required=True, help='Video file path (mp4) or camera index (0, 1, ...)')
    parser.add_argument('-c', '--classes', default=None, help='Classes file path (one class per line)')
    parser.add_argument('--score_th', type=float, default=0.5, help='Score threshold')
    parser.add_argument('--nms_th', type=float, default=0.65, help='NMS threshold')
    parser.add_argument('-o', '--output', default=None, help='Output video path')
    parser.add_argument('--cpu', action='store_true', help='Use CPU only')
    args = parser.parse_args()

    # プロバイダ設定
    if args.cpu:
        providers = ['CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    # モデル読み込み
    model = DAMOYOLO(args.model, providers=providers)

    # クラス名読み込み
    if args.classes:
        with open(args.classes, 'rt') as f:
            class_names = f.read().rstrip('\n').split('\n')
    else:
        class_names = ['class_0', 'class_1']

    # ビデオ入力
    if args.video.isdigit():
        cap = cv2.VideoCapture(int(args.video))
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f'Error: Cannot open video {args.video}')
        return

    # 出力設定
    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 推論
        bboxes, scores, class_ids = model(frame, score_th=args.score_th, nms_th=args.nms_th)

        # 描画
        result_frame = model.draw(
            frame,
            args.score_th,
            bboxes,
            scores,
            class_ids,
            class_names,
        )

        # 出力
        if writer:
            writer.write(result_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f'Processed {frame_count} frames, detections: {len(bboxes)}')

        # 表示
        cv2.imshow('DAMO-YOLO', result_frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f'Done. Total frames: {frame_count}')


if __name__ == '__main__':
    main()
