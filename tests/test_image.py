import json
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import math

# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_ID_TO_TEST = 2  # Change this to test different images
OVERLAP_RATIO_THRESHOLD = 0.3  # Same as in prepare_midog_dataset.py

def load_midog_data():
    """Load MIDOG++ JSON data"""
    with open("../MIDOG++.json", 'r') as f:
        midog_data = json.load(f)
    return midog_data

def find_image_data(midog_data, image_id):
    """Find image with specified ID"""
    for img in midog_data['images']:
        if img['id'] == image_id:
            return img
    return None

def find_annotations_for_image(midog_data, image_id):
    """Find all annotations for specified image"""
    annotations = []
    for ann in midog_data['annotations']:
        if ann['image_id'] == image_id:
            annotations.append(ann)
    return annotations

def create_patches(img_array, patch_size=640, overlap=0):
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

def bbox_intersection(bbox1, bbox2):
    """Calculate intersection of two bboxes in [x1, y1, x2, y2] format"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x1 < x2 and y1 < y2:
        return [x1, y1, x2, y2]
    else:
        return None

def calculate_overlap_ratio(bbox1, bbox2):
    """Calculate intersection over bbox1 area ratio"""
    intersection = bbox_intersection(bbox1, bbox2)
    if intersection is None:
        return 0
    
    inter_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    
    if bbox1_area == 0:
        return 0
    
    return inter_area / bbox1_area

def analyze_annotation_in_patches(ann, patch_coords):
    """Analyze which patches contain the annotation and if it would be skipped"""
    bbox = ann['bbox']  # [x1, y1, x2, y2]
    patch_analysis = []
    
    for patch_idx, coords in enumerate(patch_coords):
        x_start, y_start, x_end, y_end = coords
        patch_bbox = [x_start, y_start, x_end, y_end]
        
        intersection = bbox_intersection(bbox, patch_bbox)
        if intersection is not None:
            overlap_ratio = calculate_overlap_ratio(bbox, patch_bbox)
            is_included = overlap_ratio > OVERLAP_RATIO_THRESHOLD
            
            patch_analysis.append({
                'patch_idx': patch_idx,
                'overlap_ratio': overlap_ratio,
                'is_included': is_included,
                'coords': coords
            })
    
    return patch_analysis

def main():
    print(f"Loading MIDOG++ data...")
    midog_data = load_midog_data()
    
    # Find specified image
    img_data = find_image_data(midog_data, IMAGE_ID_TO_TEST)
    if img_data is None:
        print(f"Image with ID {IMAGE_ID_TO_TEST} not found!")
        return
    
    print(f"Found image {IMAGE_ID_TO_TEST}: {img_data['file_name']}")
    print(f"Image dimensions: {img_data['width']} x {img_data['height']}")
    
    # Find all annotations for the image
    annotations_for_img = find_annotations_for_image(midog_data, IMAGE_ID_TO_TEST)
    
    print(f"Found {len(annotations_for_img)} annotations for image {IMAGE_ID_TO_TEST}")
    
    if not annotations_for_img:
        print("No annotations found for this image!")
        return
    
    # Find the image file
    file_name = img_data['file_name']
    images_dir = "../datasets/midog_original/images"
    
    # Try different file name variations
    possible_paths = [
        os.path.join(images_dir, file_name),
        os.path.join(images_dir, f"{os.path.splitext(file_name)[0]}.tiff"),
        os.path.join(images_dir, f"{IMAGE_ID_TO_TEST}.tiff")
    ]
    
    img_path = None
    for path in possible_paths:
        if os.path.exists(path):
            img_path = path
            break
    
    if img_path is None:
        print(f"Image file not found! Tried: {possible_paths}")
        return
    
    print(f"Loading image from: {img_path}")
    
    # Load image
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img_array = np.array(img)
    
    print(f"Actual image shape: {img_array.shape}")
    
    # Create patch coordinates (no need to create actual patches)
    print("Analyzing 640x640 patch grid...")
    patches, patch_coords = create_patches(img_array, patch_size=640, overlap=0)
    
    print(f"Total patches in grid: {len(patch_coords)}")
    
    # Analyze each annotation
    annotation_analysis = {}
    for ann in annotations_for_img:
        analysis = analyze_annotation_in_patches(ann, patch_coords)
        annotation_analysis[ann['id']] = analysis
        
        if analysis:
            print(f"\nAnnotation {ann['id']}:")
            for patch_info in analysis:
                status = "INCLUDED" if patch_info['is_included'] else "SKIPPED"
                print(f"  Patch {patch_info['patch_idx']}: overlap={patch_info['overlap_ratio']:.3f} -> {status}")
        else:
            print(f"\nAnnotation {ann['id']}: No intersection with any patch")
    
    # Create a visualization of the full image with annotations and patch grid
    print("\nCreating full image visualization...")
    full_img_pil = Image.fromarray(img_array.astype('uint8'))
    draw = ImageDraw.Draw(full_img_pil)
    
    # Draw all annotations
    for ann in annotations_for_img:
        bbox = ann['bbox']  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        ann_id = ann['id']
        
        # Get analysis for this annotation
        analysis = annotation_analysis.get(ann_id, [])
        
        # Check if annotation is skipped in any patch
        has_skipped_patches = any(not patch_info['is_included'] for patch_info in analysis)
        has_included_patches = any(patch_info['is_included'] for patch_info in analysis)
        
        if has_skipped_patches and not has_included_patches:
            # Completely skipped - red
            color = 'red'
            width = 5
            status_text = "SKIPPED"
        elif has_skipped_patches and has_included_patches:
            # Partially skipped - orange
            color = 'orange'
            width = 4
            status_text = "PARTIAL"
        else:
            # Fully included - green
            color = 'green'
            width = 3
            status_text = "INCLUDED"
        
        # Draw annotation rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        # Draw annotation ID and status
        draw.text((x1, y1-30), f"{ann_id} ({status_text})", fill=color)
        
        # Draw number inside bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        draw.text((center_x-10, center_y-10), str(ann_id), fill=color)
        
        # Draw patch information for skipped annotations
        if has_skipped_patches:
            skipped_patches = [p['patch_idx'] for p in analysis if not p['is_included']]
            included_patches = [p['patch_idx'] for p in analysis if p['is_included']]
            
            text_y_offset = 10
            if skipped_patches:
                skip_text = f"Skip: P{',P'.join(map(str, skipped_patches))}"
                draw.text((x1, y2 + text_y_offset), skip_text, fill='red')
                text_y_offset += 20
            
            if included_patches:
                include_text = f"Include: P{',P'.join(map(str, included_patches))}"
                draw.text((x1, y2 + text_y_offset), include_text, fill='green')
    
    # Draw patch grid
    for i, coords in enumerate(patch_coords):
        x_start, y_start, x_end, y_end = coords
        draw.rectangle([x_start, y_start, x_end, y_end], outline='blue', width=1)
        draw.text((x_start+5, y_start+5), f"P{i}", fill='blue')
    
    # Create output directory and save
    output_dir = f"image_{IMAGE_ID_TO_TEST}_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    full_vis_path = os.path.join(output_dir, f"image_{IMAGE_ID_TO_TEST}_full_analysis.jpg")
    full_img_pil.save(full_vis_path, "JPEG", quality=95)
    
    print(f"\nResults:")
    print(f"Total patches in grid: {len(patch_coords)}")
    print(f"Output directory: {output_dir}")
    print(f"Full image visualization: {full_vis_path}")
    
    # Summary statistics
    total_annotations = len(annotations_for_img)
    completely_skipped = 0
    partially_skipped = 0
    fully_included = 0
    
    for ann in annotations_for_img:
        analysis = annotation_analysis.get(ann['id'], [])
        has_skipped = any(not p['is_included'] for p in analysis)
        has_included = any(p['is_included'] for p in analysis)
        
        if has_skipped and not has_included:
            completely_skipped += 1
        elif has_skipped and has_included:
            partially_skipped += 1
        else:
            fully_included += 1
    
    print(f"\nAnnotation Summary:")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Fully included: {fully_included}")
    print(f"  Partially skipped: {partially_skipped}")
    print(f"  Completely skipped: {completely_skipped}")
    
    print(f"\nConfiguration used:")
    print(f"  IMAGE_ID_TO_TEST: {IMAGE_ID_TO_TEST}")
    print(f"  OVERLAP_RATIO_THRESHOLD: {OVERLAP_RATIO_THRESHOLD}")
    
    print(f"\nLegend:")
    print(f"  🟢 Green: Fully included annotations")
    print(f"  🟠 Orange: Partially skipped annotations")
    print(f"  🔴 Red: Completely skipped annotations")
    print(f"  🔵 Blue: Patch grid (P0, P1, P2, ...)")

if __name__ == "__main__":
    main() 