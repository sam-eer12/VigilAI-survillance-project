import os
import yaml
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import random
from collections import defaultdict

class CombinedDatasetTrainer:
    def __init__(self):
        self.base_dir = Path(os.getcwd())
        self.combined_dataset_dir = self.base_dir / "combined_dataset"
        
        # Combined classes from all datasets
        self.combined_classes = [
            'FireExtinguisher',  # HackByte_Dataset
            'ToolBox',           # HackByte_Dataset  
            'OxygenTank',        # HackByte_Dataset
            'fire',              # FireSmokeNEWdataset
            'smoke',             # FireSmokeNEWdataset
            'other',             # FireSmokeNEWdataset
            'violence',          # violence-detection-dataset (derived from videos)
            'non-violence'       # violence-detection-dataset (derived from videos)
        ]
        
        self.class_mapping = {cls: idx for idx, cls in enumerate(self.combined_classes)}
        
    def setup_combined_dataset(self):
        """Create combined dataset structure"""
        print("Setting up combined dataset structure...")
        
        # Create directories
        for split in ['train', 'val', 'test']:
            (self.combined_dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.combined_dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copy and process each dataset
        self._process_hackbyte_dataset()
        self._process_firesmoke_dataset()
        self._process_violence_dataset()
        
        # Create combined YAML configuration
        self._create_combined_yaml()
        
        print("Combined dataset setup completed!")
        
    def _process_hackbyte_dataset(self):
        """Process HackByte dataset"""
        print("Processing HackByte dataset...")
        
        source_dir = self.base_dir / "HackByte_Dataset" / "data"
        
        for split in ['train', 'val', 'test']:
            source_split = source_dir / split
            if not source_split.exists():
                continue
                
            images_dir = source_split / 'images'
            labels_dir = source_split / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                # Copy images
                for img_file in images_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        dest_img = self.combined_dataset_dir / split / 'images' / f"hackbyte_{img_file.name}"
                        shutil.copy2(img_file, dest_img)
                        
                        # Process corresponding label
                        label_file = labels_dir / f"{img_file.stem}.txt"
                        if label_file.exists():
                            dest_label = self.combined_dataset_dir / split / 'labels' / f"hackbyte_{img_file.stem}.txt"
                            self._copy_and_update_labels(label_file, dest_label, 
                                                       ['FireExtinguisher', 'ToolBox', 'OxygenTank'])
    
    def _process_firesmoke_dataset(self):
        """Process FireSmoke dataset"""
        print("Processing FireSmoke dataset...")
        
        source_dir = self.base_dir / "FireSmokeNEWdataset.v1i.yolov9"
        
        # Map splits
        split_mapping = {'train': 'train', 'valid': 'val', 'test': 'test'}
        
        for source_split, dest_split in split_mapping.items():
            source_split_dir = source_dir / source_split
            if not source_split_dir.exists():
                continue
                
            images_dir = source_split_dir / 'images'
            labels_dir = source_split_dir / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                # Copy images
                for img_file in images_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        dest_img = self.combined_dataset_dir / dest_split / 'images' / f"firesmoke_{img_file.name}"
                        shutil.copy2(img_file, dest_img)
                        
                        # Process corresponding label
                        label_file = labels_dir / f"{img_file.stem}.txt"
                        if label_file.exists():
                            dest_label = self.combined_dataset_dir / dest_split / 'labels' / f"firesmoke_{img_file.stem}.txt"
                            self._copy_and_update_labels(label_file, dest_label, 
                                                       ['fire', 'other', 'smoke'], offset=3)
    
    def _process_violence_dataset(self):
        """Process violence detection dataset by extracting frames"""
        print("Processing violence dataset (extracting frames from videos)...")
        
        violence_dir = self.base_dir / "violence-detection-dataset"
        
        # Read action classifications
        violent_actions = self._read_action_classes(violence_dir / "violent-action-classes.csv")
        nonviolent_actions = self._read_action_classes(violence_dir / "nonviolent-action-classes.csv")
        
        # Process violent videos
        self._extract_frames_from_videos(
            violence_dir / "violent", 
            violent_actions, 
            "violence", 
            self.class_mapping['violence']
        )
        
        # Process non-violent videos  
        self._extract_frames_from_videos(
            violence_dir / "non-violent",
            nonviolent_actions,
            "non-violence",
            self.class_mapping['non-violence']
        )
    
    def _read_action_classes(self, csv_path):
        """Read action classes from CSV"""
        actions = {}
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
                for line in lines:
                    parts = line.strip().split(';')
                    if len(parts) >= 2:
                        filename = parts[0].strip()
                        action = parts[1].strip()
                        actions[filename] = action
        return actions
    
    def _extract_frames_from_videos(self, video_dir, action_classes, label_type, class_id):
        """Extract frames from videos and create labels"""
        if not video_dir.exists():
            return
            
        frame_count = 0
        max_frames_per_video = 10  # Limit frames per video to avoid overwhelming dataset
        
        for cam_dir in video_dir.glob("cam*"):
            if not cam_dir.is_dir():
                continue
                
            for video_file in cam_dir.glob("*.mp4"):
                if frame_count >= 1000:  # Limit total frames
                    break
                    
                print(f"Processing {video_file.name}...")
                
                cap = cv2.VideoCapture(str(video_file))
                if not cap.isOpened():
                    continue
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames_to_extract = min(max_frames_per_video, max(1, total_frames // 10))
                
                frame_indices = np.linspace(0, total_frames - 1, frames_to_extract, dtype=int)
                
                for i, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Determine split (80% train, 15% val, 5% test)
                        rand_val = random.random()
                        if rand_val < 0.8:
                            split = 'train'
                        elif rand_val < 0.95:
                            split = 'val'
                        else:
                            split = 'test'
                        
                        # Save frame
                        frame_name = f"violence_{cam_dir.name}_{video_file.stem}_f{i}.jpg"
                        frame_path = self.combined_dataset_dir / split / 'images' / frame_name
                        cv2.imwrite(str(frame_path), frame)
                        
                        # Create label (full frame detection for action classification)
                        label_path = self.combined_dataset_dir / split / 'labels' / f"violence_{cam_dir.name}_{video_file.stem}_f{i}.txt"
                        with open(label_path, 'w') as f:
                            # Full frame bounding box for action classification
                            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                        
                        frame_count += 1
                
                cap.release()
                
        print(f"Extracted {frame_count} frames from {label_type} videos")
    
    def _copy_and_update_labels(self, source_label, dest_label, original_classes, offset=0):
        """Copy labels and update class IDs"""
        with open(source_label, 'r') as f:
            lines = f.readlines()
        
        with open(dest_label, 'w') as f:
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_class_id = int(parts[0])
                    if old_class_id < len(original_classes):
                        # Map to combined class ID
                        class_name = original_classes[old_class_id]
                        new_class_id = self.class_mapping[class_name]
                        parts[0] = str(new_class_id)
                        f.write(' '.join(parts) + '\n')
    
    def _create_combined_yaml(self):
        """Create YAML configuration for combined dataset"""
        yaml_path = self.combined_dataset_dir / "data.yaml"
        
        config = {
            'train': str(self.combined_dataset_dir / 'train' / 'images'),
            'val': str(self.combined_dataset_dir / 'val' / 'images'),
            'test': str(self.combined_dataset_dir / 'test' / 'images'),
            'nc': len(self.combined_classes),
            'names': self.combined_classes
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Created combined dataset configuration: {yaml_path}")
        
    def train_combined_model(self, epochs=50, img_size=640, batch_size=8, device='cpu'):
        """Train YOLOv11 medium model on combined dataset"""
        print("Starting combined model training...")
        
        # Load YOLOv11 medium model
        model = YOLO('yolo11m.pt')
        
        # Path to combined dataset config
        data_yaml = str(self.combined_dataset_dir / 'data.yaml')
        
        print(f"Training with {len(self.combined_classes)} classes: {self.combined_classes}")
        print(f"Dataset config: {data_yaml}")
        
        try:
            # Train the model
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                project='runs/train',
                name='vigilai_combined_yolov11m',
                save_period=10,
                patience=30,
                optimizer='AdamW',
                lr0=0.001,
                lrf=0.0001,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                mosaic=0.5,
                mixup=0.1,
                copy_paste=0.1,
                augment=True,
                cache=True,
                single_cls=False,
                verbose=True,
                # Additional parameters for multi-class training
                cls=0.5,  # Classification loss gain
                box=7.5,  # Box loss gain
                dfl=1.5,  # Distribution focal loss gain
            )
            
            # Get the best model path
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            print(f"\nTraining completed successfully!")
            print(f"Best model saved at: {best_model_path}")
            print(f"Training results saved in: {results.save_dir}")
            
            return str(best_model_path)
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return None

def main():
    """Main training function"""
    print("=== VigilAI Combined Dataset Training ===")
    print("This will train a YOLOv11 medium model on all three datasets:")
    print("1. HackByte_Dataset (FireExtinguisher, ToolBox, OxygenTank)")
    print("2. FireSmokeNEWdataset (fire, smoke, other)")  
    print("3. violence-detection-dataset (violence, non-violence)")
    print()
    
    # Initialize trainer
    trainer = CombinedDatasetTrainer()
    
    # Setup combined dataset
    print("Step 1: Setting up combined dataset...")
    trainer.setup_combined_dataset()
    
    # Train model
    print("\nStep 2: Training combined model...")
    model_path = trainer.train_combined_model(
        epochs=100,      # More epochs for complex multi-class training
        img_size=640,    # Higher resolution for better detection
        batch_size=4,    # Smaller batch size for CPU training
        device='cpu'
    )
    
    if model_path:
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Combined model saved at: {model_path}")
        print(f"📊 Model can detect: {trainer.combined_classes}")
        
        # Test the model
        print("\nStep 3: Testing the trained model...")
        test_model(model_path, trainer.combined_classes)
    else:
        print("\n❌ Training failed!")

def test_model(model_path, class_names):
    """Test the trained model"""
    try:
        print("Loading trained model for testing...")
        model = YOLO(model_path)
        
        # Find a test image
        test_images_dir = Path("combined_dataset/test/images")
        if test_images_dir.exists():
            test_images = list(test_images_dir.glob("*.jpg"))
            if test_images:
                test_image = test_images[0]
                print(f"Testing with image: {test_image.name}")
                
                # Run inference
                results = model(str(test_image), conf=0.25)
                
                # Display results
                for r in results:
                    print(f"Detections in {test_image.name}:")
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                            print(f"  - {class_name}: {confidence:.3f}")
                    else:
                        print("  - No objects detected")
                        
        print("✅ Model testing completed!")
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")

if __name__ == "__main__":
    main()
