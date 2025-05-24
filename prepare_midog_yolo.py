import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from PIL import Image
import random
from typing import TypedDict, List, Dict, Any

# Type definitions for MIDOG JSON structure
class MidogImage(TypedDict):
    id: int
    file_name: str
    width: int
    height: int

class MidogAnnotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    bbox: List[float]  # [x1, y1, x2, y2] in absolute pixels

class MidogCategory(TypedDict):
    id: int
    name: str

class MidogDataset(TypedDict):
    images: List[MidogImage]
    annotations: List[MidogAnnotation]
    categories: List[MidogCategory]

# Paths
MIDOG_JSON = "MIDOG++.json"
IMAGES_DIR = "datasets/midog_original/images"
OUTPUT_DIR = "datasets/midog_yolo"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")
IMG_DIR_TRAIN = os.path.join(TRAIN_DIR, "images")
LABELS_DIR_TRAIN = os.path.join(TRAIN_DIR, "labels")
IMG_DIR_VAL = os.path.join(VAL_DIR, "images")
LABELS_DIR_VAL = os.path.join(VAL_DIR, "labels")
IMG_DIR_TEST = os.path.join(TEST_DIR, "images")
LABELS_DIR_TEST = os.path.join(TEST_DIR, "labels")

# Create directories
os.makedirs(IMG_DIR_TRAIN, exist_ok=True)
os.makedirs(LABELS_DIR_TRAIN, exist_ok=True)
os.makedirs(IMG_DIR_VAL, exist_ok=True)
os.makedirs(LABELS_DIR_VAL, exist_ok=True)
os.makedirs(IMG_DIR_TEST, exist_ok=True)
os.makedirs(LABELS_DIR_TEST, exist_ok=True)

# Load MIDOG data
print("Loading MIDOG++ JSON...")
with open(MIDOG_JSON, 'r') as f:
    midog_data: MidogDataset = json.load(f)

images: List[MidogImage] = midog_data.get('images', [])
annotations: List[MidogAnnotation] = midog_data.get('annotations', [])
categories: List[MidogCategory] = midog_data.get('categories', [])

# Create a mapping from image_id to annotations
print("Organizing annotations...")
image_to_annotations = {}
for ann in annotations:
    img_id = ann['image_id']
    if img_id not in image_to_annotations:
        image_to_annotations[img_id] = []
    image_to_annotations[img_id].append(ann)

# Create a mapping from category_id to class index (0-based for YOLO)
category_id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
print(f"Categories: {category_id_to_idx}")

# Split datasets (70% train, 15% validation, 15% test)
random.seed(42)
all_image_ids = list(set(img['id'] for img in images))
random.shuffle(all_image_ids)
train_split_idx = int(len(all_image_ids) * 0.7)
val_split_idx = int(len(all_image_ids) * 0.85)
train_ids = set(all_image_ids[:train_split_idx])
val_ids = set(all_image_ids[train_split_idx:val_split_idx])
test_ids = set(all_image_ids[val_split_idx:])

print(f"Training images: {len(train_ids)}")
print(f"Validation images: {len(val_ids)}")
print(f"Test images: {len(test_ids)}")

# Function to clip values to range [0, 1]
def clip_bbox(value: float) -> float:
    return max(0, min(value, 0.999))

# Process images and create YOLO labels
print("Processing images and creating YOLO labels...")
for img_data in tqdm(images):
    img_id = img_data['id']
    file_name = img_data['file_name']
    width = img_data['width']
    height = img_data['height']
    
    # Some MIDOG filenames may be different than actual files, extract just the numerical part
    tiff_file = file_name
    if not os.path.exists(os.path.join(IMAGES_DIR, tiff_file)):
        # Try to find by number
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        numeric_part = ''.join(filter(str.isdigit, base_name))
        tiff_file = f"{numeric_part}.tiff"
    
    tiff_path = os.path.join(IMAGES_DIR, tiff_file)
    if not os.path.exists(tiff_path):
        print(f"Warning: Image {tiff_file} not found. Skipping.")
        continue
    
    # Determine if this is a training, validation, or test image
    is_train = img_id in train_ids
    is_val = img_id in val_ids
    is_test = img_id in test_ids
    
    # Set destination directories
    if is_train:
        img_output_dir = IMG_DIR_TRAIN
        labels_output_dir = LABELS_DIR_TRAIN
    elif is_val:
        img_output_dir = IMG_DIR_VAL
        labels_output_dir = LABELS_DIR_VAL
    elif is_test:
        img_output_dir = IMG_DIR_TEST
        labels_output_dir = LABELS_DIR_TEST
    else:
        print(f"Warning: Image {img_id} not found in any split. Skipping.")
        continue
    
    # Convert TIFF to JPG and save
    try:
        img = Image.open(tiff_path)
        # Convert RGBA to RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        jpg_filename = f"{os.path.splitext(os.path.basename(tiff_file))[0]}.jpg"
        jpg_path = os.path.join(img_output_dir, jpg_filename)
        img.save(jpg_path, "JPEG", quality=95)
        
        # Create YOLO labels
        annotations_for_img = image_to_annotations.get(img_id, [])
        if annotations_for_img:
            label_filename = f"{os.path.splitext(os.path.basename(tiff_file))[0]}.txt"
            label_path = os.path.join(labels_output_dir, label_filename)
            
            with open(label_path, 'w') as f:
                valid_annotations = False
                for ann in annotations_for_img:
                    category_idx = category_id_to_idx.get(ann['category_id'], 0)
                    bbox = ann['bbox']  # [x1, y1, x2, y2] in absolute pixels (corner coordinates)
                    
                    # Convert to YOLO format: [center_x, center_y, width, height] normalized
                    # bbox is [x1, y1, x2, y2] format, so we need to calculate center and dimensions
                    x_center = (bbox[0] + bbox[2]) / 2 / width
                    y_center = (bbox[1] + bbox[3]) / 2 / height
                    bbox_width = (bbox[2] - bbox[0]) / width
                    bbox_height = (bbox[3] - bbox[1]) / height
                    
                    # Clip values to valid range [0, 1]
                    x_center = clip_bbox(x_center)
                    y_center = clip_bbox(y_center)
                    bbox_width = clip_bbox(bbox_width)
                    bbox_height = clip_bbox(bbox_height)
                    
                    # Skip invalid annotations (very small or zero-sized boxes)
                    if bbox_width < 0.001 or bbox_height < 0.001:
                        continue
                    
                    # Write to file
                    f.write(f"{category_idx} {x_center} {y_center} {bbox_width} {bbox_height}\n")
                    valid_annotations = True
                
                # If no valid annotations were found, create an empty annotation with class 0
                if not valid_annotations:
                    f.write(f"0 0.5 0.5 0.1 0.1\n")
    except Exception as e:
        print(f"Error processing {tiff_file}: {str(e)}")

# Clean up old cache files if they exist
for cache_file in ['datasets/midog_yolo/train/labels.cache', 'datasets/midog_yolo/val/labels.cache', 'datasets/midog_yolo/test/labels.cache']:
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Removed old cache file: {cache_file}")

# Create dataset.yaml file
yaml_content = f"""# MIDOG dataset
train: train/images
val: val/images
test: test/images

# Classes
names:
"""

for cat in categories:
    yaml_content += f"  {category_id_to_idx[cat['id']]}: {cat['name']}\n"

with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), 'w') as f:
    f.write(yaml_content)

print("Dataset preparation complete!")
print(f"Dataset configuration saved to {os.path.join(OUTPUT_DIR, 'dataset.yaml')}") 