import json
import os
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict

def load_midog_data():
    """Load MIDOG++ JSON data"""
    with open("../MIDOG++.json", 'r') as f:
        midog_data = json.load(f)
    return midog_data

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

def create_patches_coords(width, height, patch_size=640, overlap=0):
    """Create patch coordinates without loading image"""
    patch_coords = []
    stride = patch_size - overlap
    
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            x_end = min(x + patch_size, width)
            y_end = min(y + patch_size, height)
            
            # If patch is too small, skip it
            if (x_end - x) < patch_size * 0.5 or (y_end - y) < patch_size * 0.5:
                continue
            
            patch_coords.append((x, y, x_end, y_end))
    
    return patch_coords

def count_skipped_annotations_per_image(midog_data, overlap_threshold=0.3):
    """Count skipped annotations for each image"""
    images = midog_data['images']
    annotations = midog_data['annotations']
    
    # Group annotations by image
    image_to_annotations = defaultdict(list)
    for ann in annotations:
        image_to_annotations[ann['image_id']].append(ann)
    
    # Count skipped annotations per image
    skipped_counts = {}
    
    for img in images:
        img_id = img['id']
        width = img['width']
        height = img['height']
        
        annotations_for_img = image_to_annotations.get(img_id, [])
        if not annotations_for_img:
            continue
        
        # Create patch coordinates
        patch_coords = create_patches_coords(width, height, patch_size=640, overlap=0)
        
        # Count skipped annotations for this image
        skipped_annotations = set()
        
        for ann in annotations_for_img:
            bbox = ann['bbox']  # [x1, y1, x2, y2]
            annotation_included = False
            
            # Check if annotation is included in any patch
            for coords in patch_coords:
                x_start, y_start, x_end, y_end = coords
                patch_bbox = [x_start, y_start, x_end, y_end]
                
                intersection = bbox_intersection(bbox, patch_bbox)
                if intersection is not None:
                    overlap_ratio = calculate_overlap_ratio(bbox, patch_bbox)
                    if overlap_ratio > overlap_threshold:
                        annotation_included = True
                        break
            
            if not annotation_included:
                skipped_annotations.add(ann['id'])
        
        skipped_counts[img_id] = {
            'skipped_count': len(skipped_annotations),
            'total_annotations': len(annotations_for_img),
            'skipped_ids': list(skipped_annotations),
            'image_data': img
        }
    
    return skipped_counts

def visualize_image_with_annotations(img_data, annotations_for_img, skipped_annotation_ids, images_dir):
    """Visualize image with all annotations, highlighting skipped ones"""
    file_name = img_data['file_name']
    
    # Try different file name variations
    possible_paths = [
        os.path.join(images_dir, file_name),
        os.path.join(images_dir, f"{os.path.splitext(file_name)[0]}.tiff"),
        os.path.join(images_dir, f"{img_data['id']}.tiff")
    ]
    
    img_path = None
    for path in possible_paths:
        if os.path.exists(path):
            img_path = path
            break
    
    if img_path is None:
        print(f"Image file not found! Tried: {possible_paths}")
        return None
    
    print(f"Loading image from: {img_path}")
    
    # Load image
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Create drawing context
    draw = ImageDraw.Draw(img)
    
    # Draw all annotations
    for ann in annotations_for_img:
        bbox = ann['bbox']  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        
        if ann['id'] in skipped_annotation_ids:
            # Skipped annotations in red
            draw.rectangle([x1, y1, x2, y2], outline='red', width=5)
            draw.text((x1, y1-30), f"{ann['id']} (SKIPPED)", fill='red')
            # Number inside bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            draw.text((center_x-10, center_y-10), str(ann['id']), fill='red')
        else:
            # Included annotations in green
            draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
            draw.text((x1, y1-20), f"{ann['id']}", fill='green')
            # Number inside bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            draw.text((center_x-10, center_y-10), str(ann['id']), fill='green')
    
    # Draw patch grid
    patch_coords = create_patches_coords(img_data['width'], img_data['height'], patch_size=640, overlap=0)
    for i, coords in enumerate(patch_coords):
        x_start, y_start, x_end, y_end = coords
        draw.rectangle([x_start, y_start, x_end, y_end], outline='blue', width=2)
        draw.text((x_start+5, y_start+5), f"P{i}", fill='blue')
    
    return img

def main():
    print("Loading MIDOG++ data...")
    midog_data = load_midog_data()
    
    print("Analyzing skipped annotations per image...")
    skipped_counts = count_skipped_annotations_per_image(midog_data, overlap_threshold=0.8)
    
    # Find image with most skipped annotations
    max_skipped = 0
    most_skipped_image_id = None
    
    for img_id, data in skipped_counts.items():
        if data['skipped_count'] > max_skipped:
            max_skipped = data['skipped_count']
            most_skipped_image_id = img_id
    
    if most_skipped_image_id is None:
        print("No skipped annotations found!")
        return
    
    # Get data for the most skipped image
    most_skipped_data = skipped_counts[most_skipped_image_id]
    img_data = most_skipped_data['image_data']
    
    print(f"\nImage with most skipped annotations:")
    print(f"  Image ID: {most_skipped_image_id}")
    print(f"  File name: {img_data['file_name']}")
    print(f"  Dimensions: {img_data['width']} x {img_data['height']}")
    print(f"  Total annotations: {most_skipped_data['total_annotations']}")
    print(f"  Skipped annotations: {most_skipped_data['skipped_count']}")
    print(f"  Skipped annotation IDs: {most_skipped_data['skipped_ids']}")
    
    # Get all annotations for this image
    annotations_for_img = []
    for ann in midog_data['annotations']:
        if ann['image_id'] == most_skipped_image_id:
            annotations_for_img.append(ann)
    
    # Visualize the image
    images_dir = "../datasets/midog_original/images"
    result_img = visualize_image_with_annotations(
        img_data, 
        annotations_for_img, 
        most_skipped_data['skipped_ids'], 
        images_dir
    )
    
    if result_img is not None:
        # Save the visualization
        output_dir = "most_skipped_image_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"image_{most_skipped_image_id}_most_skipped.jpg")
        result_img.save(output_path, "JPEG", quality=95)
        
        print(f"\nVisualization saved to: {output_path}")
        
        # Print top 10 images with most skipped annotations
        print(f"\nTop 10 images with most skipped annotations:")
        sorted_images = sorted(skipped_counts.items(), key=lambda x: x[1]['skipped_count'], reverse=True)
        
        for i, (img_id, data) in enumerate(sorted_images[:10]):
            print(f"  {i+1}. Image {img_id}: {data['skipped_count']}/{data['total_annotations']} skipped")
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main() 