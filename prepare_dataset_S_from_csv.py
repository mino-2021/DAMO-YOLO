"""
Prepare dataset_S directly from CSV (skip YOLO txt files)
With rotation augmentation: 0, 90, 180, 270 degrees (4x data)
"""

import csv
import shutil
from pathlib import Path
import random
import json
import cv2


def rotate_bbox(x_center, y_center, width, height, rotation, img_width, img_height):
    """
    Rotate bounding box coordinates (clockwise rotation)
    rotation: 0, 90, 180, 270 degrees
    Input: normalized coordinates (0-1) relative to original image
    Returns: new (x_center, y_center, width, height) in normalized coordinates relative to rotated image

    座標変換（絶対座標での考え方）:
    - 90度CW:  new_x = orig_h - orig_y, new_y = orig_x
    - 180度:   new_x = orig_w - orig_x, new_y = orig_h - orig_y
    - 270度CW: new_x = orig_y, new_y = orig_w - orig_x
    """
    if rotation == 0:
        return x_center, y_center, width, height
    elif rotation == 90:
        # 90度時計回り: cv2.ROTATE_90_CLOCKWISE
        # 回転後の画像サイズ: (orig_h, orig_w)
        # 絶対座標: new_x = orig_h - orig_y, new_y = orig_x
        # 正規化座標: new_x_norm = (orig_h - y*orig_h) / orig_h = 1 - y
        #            new_y_norm = (x * orig_w) / orig_w = x
        new_x_center = 1 - y_center
        new_y_center = x_center
        new_width = height  # 幅と高さの比率も入れ替わる
        new_height = width
        return new_x_center, new_y_center, new_width, new_height
    elif rotation == 180:
        # 180度: (x, y) -> (1-x, 1-y)
        new_x_center = 1 - x_center
        new_y_center = 1 - y_center
        return new_x_center, new_y_center, width, height
    elif rotation == 270:
        # 270度時計回り (= 90度反時計回り): cv2.ROTATE_90_COUNTERCLOCKWISE
        # 絶対座標: new_x = orig_y, new_y = orig_w - orig_x
        # 正規化座標: new_x_norm = y, new_y_norm = 1 - x
        new_x_center = y_center
        new_y_center = 1 - x_center
        new_width = height
        new_height = width
        return new_x_center, new_y_center, new_width, new_height
    else:
        raise ValueError(f"Invalid rotation: {rotation}")


def rotate_image(img, rotation):
    """
    Rotate image by given degrees (0, 90, 180, 270)
    """
    if rotation == 0:
        return img
    elif rotation == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"Invalid rotation: {rotation}")


def prepare_dataset_S_from_csv(dataset_S_path, csv_file, output_path, train_ratio=0.8, augment_rotations=True):
    """
    Prepare dataset directly from CSV file
    Rotate images to match CSV annotation coordinates
    If augment_rotations=True, create 4x data with 0, 90, 180, 270 degree rotations
    """
    dataset_S_path = Path(dataset_S_path)
    csv_file = Path(csv_file)
    output_path = Path(output_path)

    # Read CSV and group by image
    from collections import defaultdict
    annotations_by_image = defaultdict(list)
    csv_image_sizes = {}  # Store expected image size from CSV

    print(f"Reading CSV: {csv_file}")
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row['image_path']
            x_center = float(row['x_center'])
            y_center = float(row['y_center'])
            width = float(row['width'])
            height = float(row['height'])
            csv_width = int(row['img_width'])
            csv_height = int(row['img_height'])

            # Store CSV image size
            if image_path not in csv_image_sizes:
                csv_image_sizes[image_path] = (csv_width, csv_height)

            ann_tuple = (x_center, y_center, width, height)
            annotations_by_image[image_path].append({
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })

    # Remove duplicate annotations
    for image_path in annotations_by_image:
        seen = set()
        unique_anns = []
        for ann in annotations_by_image[image_path]:
            key = (round(ann['x_center'], 6), round(ann['y_center'], 6),
                   round(ann['width'], 6), round(ann['height'], 6))
            if key not in seen:
                seen.add(key)
                unique_anns.append(ann)
        annotations_by_image[image_path] = unique_anns

    print(f"Found {len(annotations_by_image)} images with annotations")

    # Get all image paths
    all_images = list(annotations_by_image.keys())

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_images)

    train_size = int(len(all_images) * train_ratio)
    train_images = all_images[:train_size]
    val_images = all_images[train_size:]

    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    if augment_rotations:
        print("Rotation augmentation enabled: 0, 90, 180, 270 degrees (4x data)")
        rotations = [0, 90, 180, 270]
    else:
        rotations = [0]

    # Create directory structure
    for split in ['train', 'val']:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)

    ann_dir = output_path / "annotations"
    ann_dir.mkdir(exist_ok=True)

    # Process each split
    for split, image_list in [('train', train_images), ('val', val_images)]:
        print(f"\nProcessing {split}...")

        coco_format = {
            "info": {
                "description": "Dataset S (from CSV) with rotation augmentation",
                "version": "1.0",
                "year": 2025,
                "date_created": "2025/12/14"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "S", "supercategory": "S"}
            ]
        }

        image_id = 0
        annotation_id = 0

        for img_rel_path in image_list:
            # Find source image
            img_path = dataset_S_path / "image" / img_rel_path

            if not img_path.exists():
                print(f"  Warning: {img_path} not found")
                continue

            # Load image with OpenCV
            img = cv2.imread(str(img_path))
            actual_height, actual_width = img.shape[:2]

            # Get expected size from CSV
            csv_width, csv_height = csv_image_sizes[img_rel_path]

            # Check if initial rotation is needed to match CSV
            if (actual_width, actual_height) != (csv_width, csv_height):
                # Need to rotate 90 degrees to match CSV
                if (actual_width, actual_height) == (csv_height, csv_width):
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    base_width, base_height = csv_width, csv_height
                else:
                    print(f"  Warning: Size mismatch for {img_path.name}")
                    print(f"    Actual: {actual_width}x{actual_height}, CSV: {csv_width}x{csv_height}")
                    continue
            else:
                base_width, base_height = actual_width, actual_height

            # Apply rotation augmentations
            for rotation in rotations:
                # Rotate image
                rotated_img = rotate_image(img, rotation)

                # Get rotated image dimensions
                if rotation in [90, 270]:
                    img_width, img_height = base_height, base_width
                else:
                    img_width, img_height = base_width, base_height

                # Create filename with rotation suffix
                stem = img_path.stem
                suffix = img_path.suffix
                if rotation == 0:
                    new_filename = f"{stem}{suffix}"
                else:
                    new_filename = f"{stem}_rot{rotation}{suffix}"

                # Save rotated image
                dest_img = output_path / "images" / split / new_filename
                cv2.imwrite(str(dest_img), rotated_img)

                image_id += 1

                coco_format["images"].append({
                    "id": image_id,
                    "file_name": str(Path(split) / new_filename),
                    "width": img_width,
                    "height": img_height
                })

                # Add annotations with rotated coordinates
                for ann in annotations_by_image[img_rel_path]:
                    # Rotate bbox coordinates
                    new_x, new_y, new_w, new_h = rotate_bbox(
                        ann['x_center'], ann['y_center'],
                        ann['width'], ann['height'],
                        rotation, base_width, base_height
                    )

                    # Convert to absolute coordinates
                    abs_x_center = new_x * img_width
                    abs_y_center = new_y * img_height
                    abs_width = new_w * img_width
                    abs_height = new_h * img_height

                    x_min = abs_x_center - abs_width / 2
                    y_min = abs_y_center - abs_height / 2

                    annotation_id += 1

                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,  # S class
                        "bbox": [x_min, y_min, abs_width, abs_height],
                        "area": abs_width * abs_height,
                        "iscrowd": 0,
                        "segmentation": []
                    })

        # Save COCO JSON
        output_json = ann_dir / f"{split}.json"
        with open(output_json, 'w') as f:
            json.dump(coco_format, f, indent=2)

        print(f"  Images: {len(coco_format['images'])}")
        print(f"  Annotations: {len(coco_format['annotations'])}")
        print(f"  Saved to: {output_json}")

if __name__ == "__main__":
    prepare_dataset_S_from_csv(
        dataset_S_path="./datasets/dataset_S",
        csv_file="./datasets/dataset_S/yolo_annotations_S.csv",
        output_path="./datasets/dataset_S_coco",
        train_ratio=0.8,
        augment_rotations=True  # 4x data with rotations
    )
    print("\nDataset S preparation from CSV complete!")
