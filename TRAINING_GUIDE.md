# DAMO-YOLO 訓練ガイド

## 重要な注意事項

### 画像の回転問題

**問題**: torchvisionのCocoDetectionはPILで画像を読み込むため、EXIF回転情報が自動的に適用される。しかし、COCO JSONのアノテーション座標は元の画像サイズを基準にしている場合、訓練時に座標のずれが発生する。

**解決策**: **画像を事前に回転させて保存し、アノテーションもそれに合わせる**

- CSVアノテーションファイルには `img_width` と `img_height` が記録されている
- この値と実際の画像サイズを比較し、必要に応じて画像を90度回転させる
- 回転後の画像サイズとアノテーション座標を一致させる

## データセット準備手順

### 1. 単一クラスデータセット (dataset_S)

#### 準備スクリプト: `prepare_dataset_S_from_csv.py`

```bash
python prepare_dataset_S_from_csv.py
```

**処理内容**:
1. CSV (`datasets/dataset_S/yolo_annotations_S.csv`) を読み込み
2. CSVから画像の期待サイズ (`img_width`, `img_height`) を取得
3. 実際の画像を読み込み (OpenCV使用)
4. サイズが一致しない場合は画像を90度回転
5. 回転後の画像を保存
6. COCO JSON形式でアノテーションを保存

**出力**:
- `datasets/dataset_S_coco/images/train/` - 訓練画像
- `datasets/dataset_S_coco/images/val/` - 検証画像
- `datasets/dataset_S_coco/annotations/train.json` - 訓練アノテーション
- `datasets/dataset_S_coco/annotations/val.json` - 検証アノテーション

### 2. 2クラスデータセット (dataset_2class)

#### 準備スクリプト: `prepare_dataset_2class_from_csv.py`

```bash
python prepare_dataset_2class_from_csv.py
```

**処理内容**:
1. S用CSV (`datasets/dataset_S/yolo_annotations_S.csv`) を読み込み (class 0)
2. L用CSV (`datasets/dataset_L/yolo_annotations_L.csv`) を読み込み (class 1)
3. 各画像についてCSVの期待サイズと実際のサイズを比較
4. サイズが一致しない場合は画像を90度回転
5. 回転後の画像を保存
6. COCO JSON形式でアノテーションを保存 (category_id: S=1, L=2)

**出力**:
- `datasets/dataset_2class/images/train/` - 訓練画像
- `datasets/dataset_2class/images/val/` - 検証画像
- `datasets/dataset_2class/annotations/train.json` - 訓練アノテーション
- `datasets/dataset_2class/annotations/val.json` - 検証アノテーション

## 訓練手順

### 1. dataset_S の訓練

```bash
# データセット準備
rm -rf datasets/dataset_S_coco
python prepare_dataset_S_from_csv.py

# 訓練実行
python train_single_gpu.py -f configs/damoyolo_tinynasL20_T_S.py
```

**設定ファイル**: `configs/damoyolo_tinynasL20_T_S.py`
- num_classes: 1
- dataset: S_train_coco, S_val_coco
- class_names: ['S']

**訓練結果の保存場所**: `./workdirs/damoyolo_tinynasL20_T_S/`

### 2. dataset_2class の訓練

```bash
# データセット準備
rm -rf datasets/dataset_2class
python prepare_dataset_2class_from_csv.py

# 訓練実行
python train_single_gpu.py -f configs/damoyolo_tinynasL20_T_2class.py
```

**設定ファイル**: `configs/damoyolo_tinynasL20_T_2class.py`
- num_classes: 2
- dataset: 2class_train_coco, 2class_val_coco
- class_names: ['S', 'L']

**訓練結果の保存場所**: `./workdirs/damoyolo_tinynasL20_T_2class/`

## モデルの保存

訓練完了後、モデルを `saved_models/` にコピー:

### dataset_S の場合

```bash
mkdir -p saved_models/1class_S
cp ./workdirs/damoyolo_tinynasL20_T_S/latest_ckpt.pth saved_models/1class_S/damoyolo_1class_S.pth
cp configs/damoyolo_tinynasL20_T_S.py saved_models/1class_S/config.py
```

### dataset_2class の場合

```bash
mkdir -p saved_models/2class_SL
cp ./workdirs/damoyolo_tinynasL20_T_2class/latest_ckpt.pth saved_models/2class_SL/damoyolo_2class_SL.pth
cp configs/damoyolo_tinynasL20_T_2class.py saved_models/2class_SL/config.py
```

## 検出テスト

### dataset_S モデルでの検出

```bash
python tools/demo.py image \
  -f saved_models/1class_S/config.py \
  --engine saved_models/1class_S/damoyolo_1class_S.pth \
  --conf 0.5 \
  --infer_size 640 640 \
  --device cuda \
  --path datasets/dataset_S/image/OK/20251112_111906.JPG
```

### dataset_2class モデルでの検出

```bash
python tools/demo.py image \
  -f saved_models/2class_SL/config.py \
  --engine saved_models/2class_SL/damoyolo_2class_SL.pth \
  --conf 0.5 \
  --infer_size 640 640 \
  --device cuda \
  --path <画像パス>
```

**検出結果**: `./demo/` ディレクトリに保存される

## データセット検証

訓練前にアノテーションが正しいか確認:

```bash
python -c "
import cv2
import json
from pathlib import Path

# COCO JSON読み込み
data = json.load(open('datasets/dataset_S_coco/annotations/train.json'))

# 最初の画像を取得
img_info = data['images'][0]
img_path = Path('datasets/dataset_S_coco/images') / img_info['file_name']

# 画像読み込み
img = cv2.imread(str(img_path))
print(f'Image: {img_info[\"file_name\"]}')
print(f'COCO size: {img_info[\"width\"]}x{img_info[\"height\"]}')
print(f'Actual size: {img.shape[1]}x{img.shape[0]}')

# アノテーション取得
anns = [a for a in data['annotations'] if a['image_id'] == img_info['id']]
print(f'Annotations: {len(anns)}')

# バウンディングボックス描画
for ann in anns:
    bbox = ann['bbox']
    x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 0, 255), 5)

cv2.imwrite('dataset_verify.jpg', img)
print('Saved to: dataset_verify.jpg')
"
```

## トラブルシューティング

### AP (Average Precision) が非常に低い (< 10%)

**原因**: 画像とアノテーションの座標系が一致していない

**確認方法**:
1. COCO JSONの画像サイズと実際の画像サイズを比較
2. アノテーションを可視化して確認

**解決方法**:
- データセット準備スクリプトで画像を回転させる処理が正しく動作しているか確認
- CSVの `img_width`, `img_height` が正しいか確認

### class_id が間違っている

**原因**: 元のアノテーションファイルのclass_idが期待値と異なる

**確認方法**:
```bash
cat datasets/dataset_S/yolo_labels_S/20251112_111906.txt
```

**解決方法**: CSVから直接COCO JSONを作成する（YOLOフォーマットのtxtファイルは使用しない）

## ファイル構成

```
damo-yolo/
├── datasets/
│   ├── dataset_S/
│   │   ├── image/
│   │   │   ├── NG/
│   │   │   ├── OK/
│   │   │   └── White/
│   │   └── yolo_annotations_S.csv
│   ├── dataset_L/
│   │   ├── image/
│   │   │   ├── NG/
│   │   │   └── OK/
│   │   └── yolo_annotations_L.csv
│   ├── dataset_S_coco/          # 生成される
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── annotations/
│   │       ├── train.json
│   │       └── val.json
│   └── dataset_2class/          # 生成される
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── annotations/
│           ├── train.json
│           └── val.json
├── configs/
│   ├── damoyolo_tinynasL20_T_S.py
│   └── damoyolo_tinynasL20_T_2class.py
├── saved_models/
│   ├── 1class_S/
│   │   ├── config.py
│   │   └── damoyolo_1class_S.pth
│   └── 2class_SL/
│       ├── config.py
│       └── damoyolo_2class_SL.pth
├── workdirs/                    # 訓練中に生成
│   ├── damoyolo_tinynasL20_T_S/
│   └── damoyolo_tinynasL20_T_2class/
├── prepare_dataset_S_from_csv.py
├── prepare_dataset_2class_from_csv.py
└── train_single_gpu.py
```

## 期待される訓練結果

### dataset_S (1クラス)
- AP: 60-70%
- AP50: 85-95%
- AP75: 45-55%

### dataset_2class (2クラス)
- 各クラスの検出精度による
- 全体のAPは単一クラスより若干低くなる傾向
