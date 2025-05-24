import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from PIL import Image
import random
import math
from typing import TypedDict, List, Dict, Any, Tuple

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
OUTPUT_DIR = "datasets/midog_yolo_patches_640_single_class"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")
IMG_DIR_TRAIN = os.path.join(TRAIN_DIR, "images")
LABELS_DIR_TRAIN = os.path.join(TRAIN_DIR, "labels")
IMG_DIR_VAL = os.path.join(VAL_DIR, "images")
LABELS_DIR_VAL = os.path.join(VAL_DIR, "labels")
IMG_DIR_TEST = os.path.join(TEST_DIR, "images")
LABELS_DIR_TEST = os.path.join(TEST_DIR, "labels")

# Patch settings
PATCH_SIZE = 640
OVERLAP = 0  # No overlap for now, can be adjusted

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

# For single class detection, we map all categories to class 0
print("Using single class detection - all objects will be class 0")

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

def clip_bbox(value: float) -> float:
    """Clip values to range [0, 1]"""
    return max(0, min(value, 0.999))

def bbox_intersection(bbox1: List[float], bbox2: List[float]) -> List[float] | None:
    """Calculate intersection of two bboxes in [x1, y1, x2, y2] format"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x1 < x2 and y1 < y2:
        return [x1, y1, x2, y2]
    else:
        return None

def calculate_overlap_ratio(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate intersection over bbox1 area ratio"""
    intersection = bbox_intersection(bbox1, bbox2)
    if intersection is None:
        return 0
    
    inter_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    
    if bbox1_area == 0:
        return 0
    
    return inter_area / bbox1_area

def create_patches(img_array: np.ndarray, patch_size: int = 640, overlap: int = 0) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """Create patches from image array"""
    height, width = img_array.shape[:2]
    patches = []
    patch_coords = []
    
    stride = patch_size - overlap
    
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Calculate patch boundaries
            x_end = min(x + patch_size, width)
            y_end = min(y + patch_size, height)
            
            # If patch is too small, skip it
            if (x_end - x) < patch_size * 0.5 or (y_end - y) < patch_size * 0.5:
                continue
            
            # Extract patch
            patch = img_array[y:y_end, x:x_end]
            
            # Pad patch if necessary to reach exact patch_size
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded_patch
            
            patches.append(patch)
            patch_coords.append((x, y, x_end, y_end))
    
    return patches, patch_coords

# Process images and create patches
print("Processing images and creating 640x640 patches (single class, no empty patches)...")
patch_counter = 0
skipped_empty_patches = 0

for img_data in tqdm(images):
    img_id = img_data['id']
    file_name = img_data['file_name']
    width = img_data['width']
    height = img_data['height']
    
    # Find TIFF file
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
    
    try:
        # Load image
        img = Image.open(tiff_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img_array = np.array(img)
        
        # Get annotations for this image
        annotations_for_img = image_to_annotations.get(img_id, [])
        
        # Create patches
        patches, patch_coords = create_patches(img_array, PATCH_SIZE, OVERLAP)
        
        # Process each patch
        for patch_idx, (patch, (x_start, y_start, x_end, y_end)) in enumerate(zip(patches, patch_coords)):
            
            # Check if this patch has any annotations BEFORE creating files
            patch_has_annotations = False
            patch_annotations = []
            
            for ann in annotations_for_img:
                # Original bbox coordinates in full image
                bbox = ann['bbox']  # [x1, y1, x2, y2]
                
                # Check if bbox intersects with current patch
                patch_bbox = [x_start, y_start, x_end, y_end]
                intersection = bbox_intersection(bbox, patch_bbox)
                
                if intersection is not None:
                    # Calculate overlap ratio to decide if annotation should be included
                    overlap_ratio = calculate_overlap_ratio(bbox, patch_bbox)
                    
                    # Include annotation if significant overlap (>30%)
                    if overlap_ratio > 0.3:
                        # Convert intersection coordinates to patch coordinate system
                        patch_x1 = intersection[0] - x_start
                        patch_y1 = intersection[1] - y_start
                        patch_x2 = intersection[2] - x_start
                        patch_y2 = intersection[3] - y_start
                        
                        # Convert to YOLO format (normalized center coordinates and dimensions)
                        x_center = (patch_x1 + patch_x2) / 2 / PATCH_SIZE
                        y_center = (patch_y1 + patch_y2) / 2 / PATCH_SIZE
                        bbox_width = (patch_x2 - patch_x1) / PATCH_SIZE
                        bbox_height = (patch_y2 - patch_y1) / PATCH_SIZE
                        
                        # Clip values to valid range [0, 1]
                        x_center = clip_bbox(x_center)
                        y_center = clip_bbox(y_center)
                        bbox_width = clip_bbox(bbox_width)
                        bbox_height = clip_bbox(bbox_height)
                        
                        # Skip very small annotations
                        if bbox_width >= 0.01 and bbox_height >= 0.01:
                            patch_annotations.append((x_center, y_center, bbox_width, bbox_height))
                            patch_has_annotations = True
            
            # Only save patch if it has annotations
            if patch_has_annotations:
                patch_counter += 1
                
                # Create unique patch filename
                base_filename = os.path.splitext(os.path.basename(tiff_file))[0]
                patch_filename = f"{base_filename}_patch_{patch_idx:04d}"
                
                # Save patch as JPG
                patch_img = Image.fromarray(patch.astype('uint8'))
                jpg_path = os.path.join(img_output_dir, f"{patch_filename}.jpg")
                patch_img.save(jpg_path, "JPEG", quality=95)
                
                # Create YOLO labels for this patch
                label_path = os.path.join(labels_output_dir, f"{patch_filename}.txt")
                
                with open(label_path, 'w') as f:
                    for x_center, y_center, bbox_width, bbox_height in patch_annotations:
                        # All objects are class 0 (single class detection)
                        f.write(f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n")
            else:
                skipped_empty_patches += 1
        
    except Exception as e:
        print(f"Error processing {tiff_file}: {str(e)}")

# Clean up old cache files if they exist
for cache_file in ['datasets/midog_yolo_patches_640_single_class/train/labels.cache', 
                   'datasets/midog_yolo_patches_640_single_class/val/labels.cache',
                   'datasets/midog_yolo_patches_640_single_class/test/labels.cache']:
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Removed old cache file: {cache_file}")

# Create dataset.yaml file
yaml_content = f"""# MIDOG 640x640 Patches Single Class Dataset
train: train/images
val: val/images
test: test/images

# Classes (Single class - all mitotic figures)
names:
  0: mitotic_figure
"""

with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), 'w') as f:
    f.write(yaml_content)

print("640x640 Single Class Patch dataset preparation complete!")
print(f"Total patches created: {patch_counter}")
print(f"Empty patches skipped: {skipped_empty_patches}")
print(f"Dataset configuration saved to {os.path.join(OUTPUT_DIR, 'dataset.yaml')}") 