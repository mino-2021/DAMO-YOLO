"""
Prepare dataset_S directly from CSV (skip YOLO txt files)
No rotation augmentation
"""

import csv
import shutil
from pathlib import Path
import random
import json
import cv2
from collections import defaultdict


def prepare_dataset_S_from_csv(dataset_S_path, csv_file, output_path, train_ratio=0.8):
    """
    Prepare dataset directly from CSV file
    Rotate images to match CSV annotation coordinates (no augmentation)
    """
    dataset_S_path = Path(dataset_S_path)
    csv_file = Path(csv_file)
    output_path = Path(output_path)

    # Read CSV and group by image
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
                "description": "Dataset S (from CSV) - no rotation augmentation",
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
                    img_width, img_height = csv_width, csv_height
                else:
                    print(f"  Warning: Size mismatch for {img_path.name}")
                    print(f"    Actual: {actual_width}x{actual_height}, CSV: {csv_width}x{csv_height}")
                    continue
            else:
                img_width, img_height = actual_width, actual_height

            # Save image (no rotation augmentation)
            stem = img_path.stem
            suffix = img_path.suffix
            new_filename = f"{stem}{suffix}"

            dest_img = output_path / "images" / split / new_filename
            cv2.imwrite(str(dest_img), img)

            image_id += 1

            coco_format["images"].append({
                "id": image_id,
                "file_name": str(Path(split) / new_filename),
                "width": img_width,
                "height": img_height
            })

            # Add annotations
            for ann in annotations_by_image[img_rel_path]:
                # Convert to absolute coordinates
                abs_x_center = ann['x_center'] * img_width
                abs_y_center = ann['y_center'] * img_height
                abs_width = ann['width'] * img_width
                abs_height = ann['height'] * img_height

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
        output_path="./datasets/dataset_S_coco_norot",
        train_ratio=0.8
    )
    print("\nDataset S preparation (no rotation) complete!")
