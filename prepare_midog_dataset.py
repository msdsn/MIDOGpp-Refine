import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from PIL import Image
import random
import math
import argparse
import yaml
from typing import TypedDict, List, Dict, Any, Tuple

# ==========================================
# DEFAULT CONFIGURATION PARAMETERS
# ==========================================

DEFAULT_CONFIG = {
    # Patch settings
    'USE_PATCHES': True,                 # True: create patches, False: use full images
    'PATCH_SIZE': 640,                   # Only used if USE_PATCHES is True (e.g., 480, 640)
    'OVERLAP': 0,                        # Patch overlap in pixels (neighboring patches overlap)
    'ALLOW_PARTIAL_PATCHES': True,        # YENİ: varsayılan True (eski davranış)
    
    # Class settings  
    'SINGLE_CLASS': True,                # True: single class (all objects -> class 0), False: multi-class
    
    # Dataset split settings
    'INCLUDE_TEST_SET': True,            # True: train/val/test split, False: train/val split
    'TRAIN_RATIO': 0.7,                  # Training set ratio (only if INCLUDE_TEST_SET is True)
    'VAL_RATIO': 0.15,                   # Validation set ratio (only if INCLUDE_TEST_SET is True)
    # Test ratio will be 1 - TRAIN_RATIO - VAL_RATIO
    
    # When INCLUDE_TEST_SET is False, train/val split will be 80/20
    
    # Patch filtering settings (only used if USE_PATCHES is True)
    'SKIP_EMPTY_PATCHES': True,          # True: skip patches without annotations, False: save all patches
    'OVERLAP_RATIO_THRESHOLD': 0.3,      # Minimum overlap ratio to include annotation in patch (annotation area overlap)
    
    # File paths
    'MIDOG_JSON': "MIDOG++.json",
    'IMAGES_DIR': "datasets/midog_original/images",
}

# NOTE: OVERLAP vs OVERLAP_RATIO_THRESHOLD
# - OVERLAP: Physical pixel overlap between adjacent patches (e.g., 160 pixels)
# - OVERLAP_RATIO_THRESHOLD: Percentage of annotation that must be inside patch to include it (e.g., 0.3 = 30%)
#
# VISUAL EXPLANATION:
#
# 1. PATCH OVERLAP (OVERLAP parameter):
# Original image: [===============================================]
# OVERLAP = 0:    [640px---] [640px---] [640px---]  (no overlap)
# OVERLAP = 160:  [640px---]           
#                      [640px---]      
#                           [640px---]              (160px overlap)
#
# 2. ANNOTATION OVERLAP (OVERLAP_RATIO_THRESHOLD parameter):
# Patch boundary:           [============ PATCH ============]
# Annotation A:                 [--annotation--]           # 100% inside -> include
# Annotation B:       [----annotation----]              # ~60% inside -> include if threshold < 0.6
# Annotation C: [--annotation--]                     # ~20% inside -> skip if threshold > 0.2

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file or use defaults"""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Update config with user values
        config.update(user_config)
        print("Custom configuration loaded successfully!")
    else:
        if config_path:
            print(f"Config file {config_path} not found. Using default configuration.")
        else:
            print("No config file specified. Using default configuration.")
    
    return config



# ==========================================
# AUTO-GENERATED PATHS BASED ON CONFIG
# ==========================================

def generate_output_dir(config: Dict[str, Any]) -> str:
    """Generate output directory name based on configuration"""
    parts = ["midog_yolo"]
    
    if config['USE_PATCHES']:
        parts.append(f"patches_{config['PATCH_SIZE']}")
    
    if config['SINGLE_CLASS']:
        parts.append("single_class")
    else:
        parts.append("multi_class")
    
    if not config['INCLUDE_TEST_SET']:
        parts.append("train_val_only")
    
    if config['USE_PATCHES'] and config['SKIP_EMPTY_PATCHES']:
        parts.append("no_empty")
    
    return "datasets/"+"_".join(parts)

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

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare MIDOG dataset for YOLO training')
    parser.add_argument('--config', '-c', type=str, help='Path to configuration YAML file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract config values
    USE_PATCHES = config['USE_PATCHES']
    PATCH_SIZE = config['PATCH_SIZE']
    OVERLAP = config['OVERLAP']
    ALLOW_PARTIAL_PATCHES = config.get('ALLOW_PARTIAL_PATCHES', True)
    SINGLE_CLASS = config['SINGLE_CLASS']
    INCLUDE_TEST_SET = config['INCLUDE_TEST_SET']
    TRAIN_RATIO = config['TRAIN_RATIO']
    VAL_RATIO = config['VAL_RATIO']
    SKIP_EMPTY_PATCHES = config['SKIP_EMPTY_PATCHES']
    OVERLAP_RATIO_THRESHOLD = config['OVERLAP_RATIO_THRESHOLD']
    MIDOG_JSON = config['MIDOG_JSON']
    IMAGES_DIR = config['IMAGES_DIR']
    
    OUTPUT_DIR = generate_output_dir(config)
    
    print(f"\nConfiguration:")
    print(f"  USE_PATCHES: {USE_PATCHES}")
    if USE_PATCHES:
        print(f"  PATCH_SIZE: {PATCH_SIZE}")
        print(f"  OVERLAP: {OVERLAP}")
        print(f"  ALLOW_PARTIAL_PATCHES: {ALLOW_PARTIAL_PATCHES}")
        print(f"  SKIP_EMPTY_PATCHES: {SKIP_EMPTY_PATCHES}")
        print(f"  OVERLAP_RATIO_THRESHOLD: {OVERLAP_RATIO_THRESHOLD}")
    print(f"  SINGLE_CLASS: {SINGLE_CLASS}")
    print(f"  INCLUDE_TEST_SET: {INCLUDE_TEST_SET}")
    if INCLUDE_TEST_SET:
        print(f"  TRAIN_RATIO: {TRAIN_RATIO}")
        print(f"  VAL_RATIO: {VAL_RATIO}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print()

    # Create directory structure
    TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
    VAL_DIR = os.path.join(OUTPUT_DIR, "val")
    IMG_DIR_TRAIN = os.path.join(TRAIN_DIR, "images")
    LABELS_DIR_TRAIN = os.path.join(TRAIN_DIR, "labels")
    IMG_DIR_VAL = os.path.join(VAL_DIR, "images")
    LABELS_DIR_VAL = os.path.join(VAL_DIR, "labels")

    directories_to_create = [IMG_DIR_TRAIN, LABELS_DIR_TRAIN, IMG_DIR_VAL, LABELS_DIR_VAL]

    if INCLUDE_TEST_SET:
        TEST_DIR = os.path.join(OUTPUT_DIR, "test")
        IMG_DIR_TEST = os.path.join(TEST_DIR, "images")
        LABELS_DIR_TEST = os.path.join(TEST_DIR, "labels")
        directories_to_create.extend([IMG_DIR_TEST, LABELS_DIR_TEST])

    # Create directories
    for dir_path in directories_to_create:
        os.makedirs(dir_path, exist_ok=True)

    # Load MIDOG data
    print("Loading MIDOG++ JSON...")
    if not os.path.exists(MIDOG_JSON):
        print(f"Error: MIDOG JSON file '{MIDOG_JSON}' not found!")
        return
        
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
    if SINGLE_CLASS:
        print("Using single class detection - all objects will be class 0")
        category_id_to_idx = {cat['id']: 0 for cat in categories}
    else:
        category_id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
        print(f"Using multi-class detection. Categories: {category_id_to_idx}")

    # Split datasets
    random.seed(42)
    all_image_ids = list(set(img['id'] for img in images))
    random.shuffle(all_image_ids)

    if INCLUDE_TEST_SET:
        train_split_idx = int(len(all_image_ids) * TRAIN_RATIO)
        val_split_idx = int(len(all_image_ids) * (TRAIN_RATIO + VAL_RATIO))
        train_ids = set(all_image_ids[:train_split_idx])
        val_ids = set(all_image_ids[train_split_idx:val_split_idx])
        test_ids = set(all_image_ids[val_split_idx:])
        print(f"Training images: {len(train_ids)}")
        print(f"Validation images: {len(val_ids)}")
        print(f"Test images: {len(test_ids)}")
    else:
        split_idx = int(len(all_image_ids) * 0.8)
        train_ids = set(all_image_ids[:split_idx])
        val_ids = set(all_image_ids[split_idx:])
        test_ids = set()
        print(f"Training images: {len(train_ids)}")
        print(f"Validation images: {len(val_ids)}")

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

    def create_patches(img_array: np.ndarray, patch_size: int, overlap: int = 0, allow_partial: bool = True) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
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
                
                # Gerçek patch boyutları
                actual_width = x_end - x
                actual_height = y_end - y
                
                if not allow_partial:
                    # Tam boyutlu değilse atla
                    if actual_width != patch_size or actual_height != patch_size:
                        continue
                else:
                    # Eski mantık: çok küçükse atla
                    if actual_width < patch_size * 0.5 or actual_height < patch_size * 0.5:
                        continue
                
                # Extract patch
                patch = img_array[y:y_end, x:x_end]
                
                # Padding sadece allow_partial=True iken yapılır
                if allow_partial and (patch.shape[0] < patch_size or patch.shape[1] < patch_size):
                    padded_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                    padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded_patch
                
                patches.append(patch)
                patch_coords.append((x, y, x_end, y_end))
        
        return patches, patch_coords

    def get_output_directories(img_id: int):
        """Get output directories based on image split"""
        if img_id in train_ids:
            return IMG_DIR_TRAIN, LABELS_DIR_TRAIN
        elif img_id in val_ids:
            return IMG_DIR_VAL, LABELS_DIR_VAL
        elif INCLUDE_TEST_SET and img_id in test_ids:
            return IMG_DIR_TEST, LABELS_DIR_TEST
        else:
            return None, None

    def process_annotations_for_patch(annotations_for_img: List[MidogAnnotation], 
                                      patch_coords: Tuple[int, int, int, int],
                                      patch_size: int) -> List[Tuple[int, float, float, float, float]]:
        """Process annotations for a specific patch"""
        x_start, y_start, x_end, y_end = patch_coords
        patch_annotations = []
        
        for ann in annotations_for_img:
            bbox = ann['bbox']  # [x1, y1, x2, y2]
            
            # Check if bbox intersects with current patch
            patch_bbox = [x_start, y_start, x_end, y_end]
            intersection = bbox_intersection(bbox, patch_bbox)
            
            if intersection is not None:
                # Calculate overlap ratio
                overlap_ratio = calculate_overlap_ratio(bbox, patch_bbox)
                
                # Include annotation if significant overlap
                if overlap_ratio > OVERLAP_RATIO_THRESHOLD:
                    category_idx = category_id_to_idx.get(ann['category_id'], 0)
                    
                    # Convert intersection coordinates to patch coordinate system
                    patch_x1 = intersection[0] - x_start
                    patch_y1 = intersection[1] - y_start
                    patch_x2 = intersection[2] - x_start
                    patch_y2 = intersection[3] - y_start
                    
                    # Convert to YOLO format (normalized center coordinates and dimensions)
                    x_center = (patch_x1 + patch_x2) / 2 / patch_size
                    y_center = (patch_y1 + patch_y2) / 2 / patch_size
                    bbox_width = (patch_x2 - patch_x1) / patch_size
                    bbox_height = (patch_y2 - patch_y1) / patch_size
                    
                    # Clip values to valid range [0, 1]
                    x_center = clip_bbox(x_center)
                    y_center = clip_bbox(y_center)
                    bbox_width = clip_bbox(bbox_width)
                    bbox_height = clip_bbox(bbox_height)
                    
                    # Skip very small annotations
                    if bbox_width >= 0.01 and bbox_height >= 0.01:
                        patch_annotations.append((category_idx, x_center, y_center, bbox_width, bbox_height))
        
        return patch_annotations

    def process_annotations_for_full_image(annotations_for_img: List[MidogAnnotation], 
                                           width: int, height: int) -> List[Tuple[int, float, float, float, float]]:
        """Process annotations for full image"""
        full_image_annotations = []
        
        for ann in annotations_for_img:
            category_idx = category_id_to_idx.get(ann['category_id'], 0)
            bbox = ann['bbox']  # [x1, y1, x2, y2] in absolute pixels
            
            # Convert to YOLO format: [center_x, center_y, width, height] normalized
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
            if bbox_width >= 0.001 and bbox_height >= 0.001:
                full_image_annotations.append((category_idx, x_center, y_center, bbox_width, bbox_height))
        
        return full_image_annotations

    # Process images
    if USE_PATCHES:
        print(f"Processing images and creating {PATCH_SIZE}x{PATCH_SIZE} patches...")
    else:
        print("Processing full images...")

    patch_counter = 0
    skipped_empty_patches = 0
    processed_images = 0

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
        
        # Get output directories
        img_output_dir, labels_output_dir = get_output_directories(img_id)
        if img_output_dir is None:
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
            
            if USE_PATCHES:
                # Create patches
                patches, patch_coords = create_patches(img_array, PATCH_SIZE, OVERLAP, ALLOW_PARTIAL_PATCHES)
                
                # Process each patch
                for patch_idx, (patch, coords) in enumerate(zip(patches, patch_coords)):
                    
                    # Process annotations for this patch
                    patch_annotations = process_annotations_for_patch(annotations_for_img, coords, PATCH_SIZE)
                    
                    # Check if we should save this patch
                    has_annotations = len(patch_annotations) > 0
                    should_save = has_annotations or not SKIP_EMPTY_PATCHES
                    
                    if should_save:
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
                            for category_idx, x_center, y_center, bbox_width, bbox_height in patch_annotations:
                                f.write(f"{category_idx} {x_center} {y_center} {bbox_width} {bbox_height}\n")
                    else:
                        skipped_empty_patches += 1
            else:
                # Process full image
                processed_images += 1
                
                # Convert TIFF to JPG and save
                jpg_filename = f"{os.path.splitext(os.path.basename(tiff_file))[0]}.jpg"
                jpg_path = os.path.join(img_output_dir, jpg_filename)
                img.save(jpg_path, "JPEG", quality=95)
                
                # Process annotations for full image
                full_image_annotations = process_annotations_for_full_image(annotations_for_img, width, height)
                
                # Create YOLO labels
                label_filename = f"{os.path.splitext(os.path.basename(tiff_file))[0]}.txt"
                label_path = os.path.join(labels_output_dir, label_filename)
                
                with open(label_path, 'w') as f:
                    if full_image_annotations:
                        for category_idx, x_center, y_center, bbox_width, bbox_height in full_image_annotations:
                            f.write(f"{category_idx} {x_center} {y_center} {bbox_width} {bbox_height}\n")
                    else:
                        # No annotations - create empty file or dummy annotation based on configuration
                        if not SINGLE_CLASS:
                            # For multi-class, create a dummy annotation to avoid issues
                            f.write(f"0 0.5 0.5 0.1 0.1\n")
                        # For single class, empty file is fine
            
        except Exception as e:
            print(f"Error processing {tiff_file}: {str(e)}")

    # Clean up old cache files
    cache_patterns = [
        os.path.join(TRAIN_DIR, "labels.cache"),
        os.path.join(VAL_DIR, "labels.cache")
    ]
    if INCLUDE_TEST_SET:
        cache_patterns.append(os.path.join(TEST_DIR, "labels.cache"))

    for cache_file in cache_patterns:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Removed old cache file: {cache_file}")

    # Create dataset.yaml file
    yaml_content = f"""# MIDOG Dataset - Generated Configuration
train: train/images
val: val/images
"""

    if INCLUDE_TEST_SET:
        yaml_content += "test: test/images\n"

    yaml_content += "\n# Classes\nnames:\n"

    if SINGLE_CLASS:
        yaml_content += "  0: mitotic_figure\n"
    else:
        for cat in categories:
            yaml_content += f"  {category_id_to_idx[cat['id']]}: {cat['name']}\n"

    with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), 'w') as f:
        f.write(yaml_content)

    print("\nDataset preparation complete!")
    if USE_PATCHES:
        print(f"Total patches created: {patch_counter}")
        if SKIP_EMPTY_PATCHES:
            print(f"Empty patches skipped: {skipped_empty_patches}")
    else:
        print(f"Total images processed: {processed_images}")
    print(f"Dataset configuration saved to {os.path.join(OUTPUT_DIR, 'dataset.yaml')}")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 