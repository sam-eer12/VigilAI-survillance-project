import zipfile
import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import time

class VigilAI:
    def __init__(self, model_path=None, model_size='m', use_combined_model=False):
        self.model = None
        self.model_path = model_path
        self.model_size = model_size  # n, s, m, l, x for nano, small, medium, large, extra-large
        self.use_combined_model = use_combined_model
        
        # Define classes based on model type
        if use_combined_model:
            # Combined model classes from all datasets
            self.class_names = [
                'FireExtinguisher', 'ToolBox', 'OxygenTank',  # HackByte_Dataset
                'fire', 'smoke', 'other',                      # FireSmokeNEWdataset
                'violence', 'non-violence'                     # violence-detection-dataset
            ]
            self.colors = {
                'FireExtinguisher': (0, 0, 255),    # Red
                'ToolBox': (255, 0, 0),             # Blue  
                'OxygenTank': (0, 255, 0),          # Green
                'fire': (0, 69, 255),               # Orange-Red
                'smoke': (128, 128, 128),           # Gray
                'other': (255, 255, 0),             # Cyan
                'violence': (0, 0, 139),            # Dark Red
                'non-violence': (0, 255, 0)         # Green
            }
        else:
            # Original HackByte dataset classes
            self.class_names = ['FireExtinguisher', 'ToolBox', 'OxygenTank']
            self.colors = {
                'FireExtinguisher': (0, 0, 255),    # Red
                'ToolBox': (255, 0, 0),             # Blue  
                'OxygenTank': (0, 255, 0)           # Green
            }
    
    def extract_dataset(self):
        """Extract the HackByte dataset from the zip file"""
        import zipfile
        import os
        
        # Check if dataset already exists
        if os.path.exists('HackByte_Dataset'):
            print("Dataset already extracted. Skipping extraction.")
            return True
        
        # Check if zip file exists
        if not os.path.exists('Hackathon_Dataset.zip'):
            print("Error: Hackathon_Dataset.zip not found!")
            return False
        
        try:
            print("Extracting HackByte dataset...")
            with zipfile.ZipFile('Hackathon_Dataset.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            print("Dataset extraction completed!")
            return True
        except Exception as e:
            print(f"Error extracting dataset: {str(e)}")
            return False
    
    def check_datasets_available(self):
        """Check if all required datasets are available for combined training"""
        required_datasets = [
            'HackByte_Dataset',
            'FireSmokeNEWdataset.v1i.yolov9',
            'violence-detection-dataset'
        ]
        
        missing_datasets = []
        for dataset in required_datasets:
            if not os.path.exists(dataset):
                missing_datasets.append(dataset)
        
        if missing_datasets:
            print(f"Warning: Missing datasets for combined training: {missing_datasets}")
            return False
        
        print("All datasets available for combined training!")
        return True
        
            
    def train_combined_yolov11(self, epochs=50, img_size=640, batch_size=4, device='cpu'):
        """
        Train YOLOv11 model on combined dataset from all three datasets
        """
        from train_combined_model import CombinedDatasetTrainer
        
        try:
            print("Initializing combined dataset trainer...")
            trainer = CombinedDatasetTrainer()
            
            print("Setting up combined dataset...")
            trainer.setup_combined_dataset()
            
            print("Starting combined model training...")
            model_path = trainer.train_combined_model(
                epochs=epochs,
                img_size=img_size,
                batch_size=batch_size,
                device=device
            )
            
            if model_path:
                self.model_path = model_path
                self.use_combined_model = True
                self.class_names = trainer.combined_classes
                # Update colors for new classes
                self.colors.update({
                    'fire': (0, 69, 255),               # Orange-Red
                    'smoke': (128, 128, 128),           # Gray
                    'other': (255, 255, 0),             # Cyan
                    'violence': (0, 0, 139),            # Dark Red
                    'non-violence': (0, 255, 0)         # Green
                })
                print(f"Combined model training completed! Model saved at: {model_path}")
                return model_path
            else:
                print("Combined model training failed!")
                return None
                
        except Exception as e:
            print(f"Error during combined training: {str(e)}")
            return None
            
    def train_yolov11(self, epochs=20, img_size=416, batch_size=16, device='cpu'):
        """
        Train YOLOv11 model on the dataset
        """
        try:
            # Load YOLOv11 model based on size parameter
            model_file = f'yolo11{self.model_size}.pt'
            print(f"Loading YOLOv11 model: {model_file}")
            model = YOLO(model_file)  # Using specified model size for better accuracy
            
            # Update the yaml file path to be absolute
            data_yaml = os.path.join(os.getcwd(), 'HackByte_Dataset', 'yolo_params.yaml')
            
            # Ensure the yaml file has correct paths
            self._update_yaml_paths(data_yaml)
            
            print(f"Starting training with {epochs} epochs...")
            print(f"Dataset config: {data_yaml}")
            
            # Train the model
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                project='runs/train',
                name='vigilai_yolov11',
                save_period=10,  # Save checkpoint every 10 epochs
                patience=20,     # Early stopping patience
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
                verbose=True
            )
            
            # Get the best model path
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            self.model_path = str(best_model_path)
            
            print(f"Training completed! Best model saved at: {self.model_path}")
            return results
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return None
    
    def _update_yaml_paths(self, yaml_path):
        """
        Update the YAML file to use absolute paths
        """
        import yaml
        
        base_dir = os.path.dirname(yaml_path)
        
        # Read current yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Update paths to absolute paths
        data['train'] = os.path.join(base_dir, 'data', 'train', 'images')
        data['val'] = os.path.join(base_dir, 'data', 'val', 'images')
        data['test'] = os.path.join(base_dir, 'data', 'test', 'images')
        
        # Write back
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
            
        print(f"Updated YAML paths in {yaml_path}")
    
    def load_model(self, model_path=None):
        """
        Load trained model
        """
        if model_path:
            self.model_path = model_path
            
        if not self.model_path:
            print("No model path specified!")
            return False
            
        try:
            print(f"Loading model from {self.model_path}")
            self.model = YOLO(self.model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def detect_objects(self, frame, conf_threshold=0.5):
        """
        Detect objects in a frame
        """
        if self.model is None:
            print("Model not loaded!")
            return frame, []
        
        # Run inference
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        detections = []
        
        # Process results
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                        color = self.colors.get(class_name, (255, 255, 255))
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                                    (int(x1) + label_size[0], int(y1)), color, -1)
                        cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
        
        return frame, detections
    
    def live_detection(self, camera_index=0, model_path=None, conf_threshold=0.5):
        """
        Run live detection on camera feed
        """
        # Load model if not already loaded
        if self.model is None:
            if model_path:
                self.load_model(model_path)
            else:
                print("Please provide a model path or train a model first!")
                return
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        print("Starting live detection...")
        print("Controls:")
        print("  - Press 'Q' or 'ESC' to quit")
        print("  - Press 'S' to save screenshot")
        print("  - Click the STOP button on screen to quit")
        print(f"Detecting: {', '.join(self.class_names)}")
        
        frame_count = 0
        fps_start_time = time.time()
        
        # Mouse callback function for stop button
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if click is on stop button (top-right corner)
                if x >= param['stop_x'] and x <= param['stop_x'] + param['stop_w'] and \
                   y >= param['stop_y'] and y <= param['stop_y'] + param['stop_h']:
                    param['stop_clicked'] = True
        
        # Set up mouse callback
        cv2.namedWindow('VigilAI - Live Detection')
        mouse_params = {'stop_clicked': False, 'stop_x': 0, 'stop_y': 0, 'stop_w': 80, 'stop_h': 30}
        cv2.setMouseCallback('VigilAI - Live Detection', mouse_callback, mouse_params)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Detect objects
            annotated_frame, detections = self.detect_objects(frame, conf_threshold)
            
            # Print detections to terminal
            if detections:
                timestamp = time.strftime("%H:%M:%S")
                print(f"\n[{timestamp}] DETECTIONS FOUND:")
                for i, detection in enumerate(detections, 1):
                    bbox = detection['bbox']
                    print(f"  {i}. {detection['class']} - Confidence: {detection['confidence']:.3f} - BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                print(f"  Total objects detected: {len(detections)}")
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
            else:
                fps = 0
            
            # Add FPS and detection count to frame
            info_text = f"FPS: {fps:.1f} | Detections: {len(detections)}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add stop button in top-right corner
            frame_height, frame_width = annotated_frame.shape[:2]
            stop_w, stop_h = 80, 30
            stop_x = frame_width - stop_w - 10
            stop_y = 10
            
            # Update mouse params with current button position
            mouse_params['stop_x'] = stop_x
            mouse_params['stop_y'] = stop_y
            
            # Draw stop button background
            cv2.rectangle(annotated_frame, (stop_x, stop_y), (stop_x + stop_w, stop_y + stop_h), (0, 0, 255), -1)
            cv2.rectangle(annotated_frame, (stop_x, stop_y), (stop_x + stop_w, stop_y + stop_h), (255, 255, 255), 2)
            
            # Add STOP text
            text_size = cv2.getTextSize("STOP", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = stop_x + (stop_w - text_size[0]) // 2
            text_y = stop_y + (stop_h + text_size[1]) // 2
            cv2.putText(annotated_frame, "STOP", (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add controls instruction at bottom
            controls_text = "Controls: Q/ESC=Quit | S=Screenshot | Click STOP button"
            cv2.putText(annotated_frame, controls_text, (10, frame_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display detection info
            y_offset = 60
            for detection in detections:
                det_info = f"{detection['class']}: {detection['confidence']:.2f}"
                cv2.putText(annotated_frame, det_info, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
            
            # Show frame
            cv2.imshow('VigilAI - Live Detection', annotated_frame)
            
            # Handle key presses and stop button
            key = cv2.waitKey(1) & 0xFF
            
            # Check if stop button was clicked
            if mouse_params['stop_clicked']:
                print("Stop button clicked! Stopping detection...")
                break
                
            # Check keyboard inputs
            if key == ord('q') or key == ord('Q'):
                print("Q key pressed! Stopping detection...")
                break
            elif key == 27:  # ESC key
                print("ESC key pressed! Stopping detection...")
                break
            elif key == ord('s') or key == ord('S'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"detection_screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        # Cleanup
        print("Cleaning up and closing camera...")
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped successfully!")
    
    def evaluate_model(self, model_path=None):
        """
        Evaluate the trained model on test dataset
        """
        if model_path:
            self.load_model(model_path)
        
        if self.model is None:
            print("No model loaded for evaluation!")
            return None
        
        try:
            data_yaml = os.path.join(os.getcwd(), 'HackByte_Dataset', 'yolo_params.yaml')
            results = self.model.val(data=data_yaml, verbose=True)
            print("Model evaluation completed!")
            return results
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None
    
    def evaluate_combined_model(self, model_path=None):
        """
        Evaluate the combined model on combined dataset
        """
        if model_path:
            self.load_model(model_path)
        
        if self.model is None:
            print("No model loaded for evaluation!")
            return None
        
        try:
            # Use combined dataset for evaluation
            data_yaml = os.path.join(os.getcwd(), 'combined_dataset', 'data.yaml')
            if not os.path.exists(data_yaml):
                print("Combined dataset not found. Please run combined training first.")
                return None
            
            results = self.model.val(data=data_yaml, verbose=True)
            print("Combined model evaluation completed!")
            return results
        except Exception as e:
            print(f"Error during combined evaluation: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description='VigilAI Surveillance System')
    parser.add_argument('--mode', type=str, 
                       choices=['extract', 'train', 'train-combined', 'detect', 'evaluate'], 
                       default='detect', help='Mode to run: extract, train, train-combined, detect, or evaluate')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for live detection')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detection')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--model-size', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'], 
                       help='YOLOv11 model size: n(nano), s(small), m(medium), l(large), x(extra-large)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Initialize VigilAI
    vigilai = VigilAI(model_path=args.model, model_size=args.model_size)
    
    if args.mode == 'extract':
        vigilai.extract_dataset()
        
    elif args.mode == 'train':
        print("Starting YOLOv11 training on HackByte dataset...")
        results = vigilai.train_yolov11(
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch,
            device=args.device
        )
        if results:
            print("Training completed successfully!")
    
    elif args.mode == 'train-combined':
        print("Starting YOLOv11 combined training on all three datasets...")
        print("Datasets: HackByte + FireSmoke + Violence Detection")
        model_path = vigilai.train_combined_yolov11(
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch,
            device=args.device
        )
        if model_path:
            print(f"Combined training completed successfully!")
            print(f"Model saved at: {model_path}")
            print(f"Classes: {vigilai.class_names}")
        
    elif args.mode == 'train':
        print("Starting YOLOv11 training...")
        results = vigilai.train_yolov11(
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch
        )
        if results:
            print("Training completed successfully!")
            
    elif args.mode == 'detect':
        if not args.model:
            # Try to find the latest trained model
            runs_dir = Path('runs/train')
            if runs_dir.exists():
                latest_run = max(runs_dir.glob('vigilai_yolov11*'), 
                               key=lambda x: x.stat().st_mtime, default=None)
                if latest_run and (latest_run / 'weights' / 'best.pt').exists():
                    args.model = str(latest_run / 'weights' / 'best.pt')
                    print(f"Using latest trained model: {args.model}")
        
        if args.model and os.path.exists(args.model):
            vigilai.live_detection(
                camera_index=args.camera,
                model_path=args.model,
                conf_threshold=args.conf
            )
        else:
            print("No model found! Please train a model first or specify a valid model path.")
            print("Run: python vigilai.py --mode train")
            
    elif args.mode == 'evaluate':
        if args.model:
            vigilai.evaluate_model(args.model)
        else:
            print("Please specify a model path for evaluation!")

if __name__ == "__main__":
    main()

