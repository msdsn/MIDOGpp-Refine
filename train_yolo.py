import sys
import os

# Add ultralytics to path
sys.path.append('ultralytics')

from ultralytics import YOLO

def main():
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # nano model for faster training
    
    # Train the model
    results = model.train(
        data='datasets/midog_yolo_patches_640_single_class/dataset.yaml',
        epochs=100,
        imgsz=480,
        batch=16,
        name='midog_mitosis',
        save_period=10,
        patience=15,
        #device='cpu'  
    )
    
    # Save final model
    model.save('midog_final.pt')
    print("Training completed!")
    print(f"Best model saved at: runs/detect/midog_mitosis/weights/best.pt")

if __name__ == "__main__":
    main() 