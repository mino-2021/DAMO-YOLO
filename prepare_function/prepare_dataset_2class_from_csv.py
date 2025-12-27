"""
Prepare 2-class dataset directly from CSV files
With rotation augmentation: 0, 90, 180, 270 degrees (4x data)
"""

import csv
import shutil
from pathlib import Path
import random
import json
import cv2
from collections import defaultdict


def rotate_bbox(x_center, y_center, width, height, rotation):
    """
    Rotate bounding box coordinates (clockwise rotation)
    rotation: 0, 90, 180, 270 degrees
    """
    if rotation == 0:
        return x_center, y_center, width, height
    elif rotation == 90:
        new_x_center = 1 - y_center
        new_y_center = x_center
        new_width = height
        new_height = width
        return new_x_center, new_y_center, new_width, new_height
    elif rotation == 180:
        new_x_center = 1 - x_center
        new_y_center = 1 - y_center
        return new_x_center, new_y_center, width, height
    elif rotation == 270:
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


def remove_duplicate_annotations(annotations_dict):
    """Remove duplicate annotations from each image"""
    for image_path in annotations_dict:
        seen = set()
        unique_anns = []
        for ann in annotations_dict[image_path]:
            key = (round(ann['x_center'], 6), round(ann['y_center'], 6),
                   round(ann['width'], 6), round(ann['height'], 6))
            if key not in seen:
                seen.add(key)
                unique_anns.append(ann)
        annotations_dict[image_path] = unique_anns
    return annotations_dict


def prepare_2class_from_csv(dataset_L_path, dataset_S_path,
                            csv_L_file, csv_S_file,
                            output_path, train_ratio=0.8, augment_rotations=True):
    """
    Prepare 2-class dataset directly from CSV files
    - Class 0: S (from dataset_S)
    - Class 1: L (from dataset_L)
    If augment_rotations=True, create 4x data with 0, 90, 180, 270 degree rotations
    """
    dataset_L_path = Path(dataset_L_path)
    dataset_S_path = Path(dataset_S_path)
    output_path = Path(output_path)

    # Read CSV files
    all_data = []  # (image_path, annotations, class_id, csv_size)

    # Process S class (class 0)
    print("Reading S class CSV...")
    annotations_S = defaultdict(list)
    csv_sizes_S = {}

    with open(csv_S_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row['image_path']
            x_center = float(row['x_center'])
            y_center = float(row['y_center'])
            width = float(row['width'])
            height = float(row['height'])
            csv_width = int(row['img_width'])
            csv_height = int(row['img_height'])

            if image_path not in csv_sizes_S:
                csv_sizes_S[image_path] = (csv_width, csv_height)

            annotations_S[image_path].append({
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })

    # Remove duplicates
    annotations_S = remove_duplicate_annotations(annotations_S)

    for img_rel_path, anns in annotations_S.items():
        img_path = dataset_S_path / "image" / img_rel_path
        if img_path.exists():
            all_data.append((img_path, anns, 0, csv_sizes_S[img_rel_path]))

    print(f"Found {len(annotations_S)} S images")

    # Process L class (class 1)
    print("Reading L class CSV...")
    annotations_L = defaultdict(list)
    csv_sizes_L = {}

    with open(csv_L_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row['image_path']
            x_center = float(row['x_center'])
            y_center = float(row['y_center'])
            width = float(row['width'])
            height = float(row['height'])
            csv_width = int(row['img_width'])
            csv_height = int(row['img_height'])

            if image_path not in csv_sizes_L:
                csv_sizes_L[image_path] = (csv_width, csv_height)

            annotations_L[image_path].append({
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })

    # Remove duplicates
    annotations_L = remove_duplicate_annotations(annotations_L)

    l_count = 0
    for img_rel_path, anns in annotations_L.items():
        img_path = dataset_L_path / "image" / img_rel_path
        if img_path.exists():
            all_data.append((img_path, anns, 1, csv_sizes_L[img_rel_path]))
            l_count += 1

    print(f"Found {len(annotations_L)} L images")
    print(f"Total: {len(all_data)} images")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_data)

    train_size = int(len(all_data) * train_ratio)
    train_data = all_data[:train_size]
    val_data = all_data[train_size:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

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
    for split, data_list in [('train', train_data), ('val', val_data)]:
        print(f"\nProcessing {split}...")

        coco_format = {
            "info": {
                "description": "2-Class Dataset (S and L) from CSV with rotation augmentation",
                "version": "1.0",
                "year": 2025,
                "date_created": "2025/12/14"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "S", "supercategory": "S"},
                {"id": 2, "name": "L", "supercategory": "L"}
            ]
        }

        image_id = 0
        annotation_id = 0

        for img_path, anns, class_id, csv_size in data_list:
            # Load image with OpenCV
            img = cv2.imread(str(img_path))
            actual_height, actual_width = img.shape[:2]

            # Get expected size from CSV
            csv_width, csv_height = csv_size

            # Check if rotation is needed to match CSV
            if (actual_width, actual_height) != (csv_width, csv_height):
                # Need to rotate 90 degrees
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
                for ann in anns:
                    # Rotate bbox coordinates
                    new_x, new_y, new_w, new_h = rotate_bbox(
                        ann['x_center'], ann['y_center'],
                        ann['width'], ann['height'],
                        rotation
                    )

                    # Convert to absolute coordinates
                    abs_x_center = new_x * img_width
                    abs_y_center = new_y * img_height
                    abs_width = new_w * img_width
                    abs_height = new_h * img_height

                    x_min = abs_x_center - abs_width / 2
                    y_min = abs_y_center - abs_height / 2

                    annotation_id += 1

                    # COCO category_id: S=1, L=2
                    coco_category_id = class_id + 1

                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": coco_category_id,
                        "bbox": [x_min, y_min, abs_width, abs_height],
                        "area": abs_width * abs_height,
                        "iscrowd": 0,
                        "segmentation": []
                    })

        # Save COCO JSON
        output_json = ann_dir / f"{split}.json"
        with open(output_json, 'w') as f:
            json.dump(coco_format, f, indent=2)

        s_count = len([a for a in coco_format['annotations'] if a['category_id'] == 1])
        l_count = len([a for a in coco_format['annotations'] if a['category_id'] == 2])

        print(f"  Images: {len(coco_format['images'])}")
        print(f"  Annotations: {len(coco_format['annotations'])} (S: {s_count}, L: {l_count})")
        print(f"  Saved to: {output_json}")

if __name__ == "__main__":
    prepare_2class_from_csv(
        dataset_L_path="./datasets/dataset_L",
        dataset_S_path="./datasets/dataset_S",
        csv_L_file="./datasets/dataset_L/yolo_annotations_L.csv",
        csv_S_file="./datasets/dataset_S/yolo_annotations_S.csv",
        output_path="./datasets/dataset_2class",
        train_ratio=0.8,
        augment_rotations=True  # 4x data with rotations
    )
    print("\n2-class dataset preparation from CSV complete!")
