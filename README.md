# VigilAI - YOLOv11 Surveillance System

A comprehensive surveillance system using YOLOv11 for object detection with live camera feed capabilities.

## Features

- **YOLOv11 Training**: Train custom YOLOv11 models on your dataset
- **Live Detection**: Real-time object detection using webcam/camera feed
- **Multi-Class Detection**: Detects FireExtinguisher, ToolBox, and OxygenTank
- **Model Evaluation**: Comprehensive model evaluation tools
- **Easy Setup**: Automated environment setup

## Classes Detected

1. **FireExtinguisher** (Red bounding box)
2. **ToolBox** (Blue bounding box) 
3. **OxygenTank** (Green bounding box)

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Webcam/Camera for live detection

## Quick Setup

### Option 1: Automatic Setup
Run the setup script:
```bash
setup.bat
```

### Option 2: Manual Setup
```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## Usage

### 1. Extract Dataset
```bash
python vigilai.py --mode extract
```

### 2. Train YOLOv11 Model
```bash
# Basic training
python vigilai.py --mode train

# Custom training parameters
python vigilai.py --mode train --epochs 100 --batch 32 --img-size 640
```

### 3. Live Camera Detection
```bash
# Use latest trained model
python vigilai.py --mode detect

# Use specific model
python vigilai.py --mode detect --model runs/train/vigilai_yolov11/weights/best.pt

# Custom camera and confidence
python vigilai.py --mode detect --camera 0 --conf 0.6
```

### 4. Evaluate Model
```bash
python vigilai.py --mode evaluate --model runs/train/vigilai_yolov11/weights/best.pt
```

## Live Detection Controls

- **Q**: Quit the detection
- **S**: Save screenshot with timestamp

## Training Parameters

- **Epochs**: Number of training iterations (default: 50)
- **Batch Size**: Training batch size (default: 16)
- **Image Size**: Input image resolution (default: 640)
- **Confidence Threshold**: Detection confidence (default: 0.5)

## Model Architecture

The system uses YOLOv11n (nano) by default for faster inference. You can modify the code to use:
- `yolo11n.pt` - Nano (fastest)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium  
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large (most accurate)

## Output Structure

```
runs/
└── train/
    └── vigilai_yolov11/
        ├── weights/
        │   ├── best.pt      # Best model weights
        │   └── last.pt      # Last epoch weights
        ├── results.png      # Training metrics
        ├── confusion_matrix.png
        └── val_batch0_labels.jpg
```

## Advanced Usage

### Custom Training Script
```python
from vigilai import VigilAI

# Initialize
vigilai = VigilAI()

# Extract dataset
vigilai.extract_dataset()

# Train with custom parameters
results = vigilai.train_yolov11(
    epochs=100,
    img_size=640,
    batch_size=32,
    device='cuda:0'
)

# Load and use model
vigilai.load_model('path/to/best.pt')
vigilai.live_detection(camera_index=0, conf_threshold=0.6)
```

### Programmatic Detection
```python
import cv2
from vigilai import VigilAI

vigilai = VigilAI()
vigilai.load_model('path/to/model.pt')

# Process single frame
frame = cv2.imread('image.jpg')
annotated_frame, detections = vigilai.detect_objects(frame)

# Print detections
for det in detections:
    print(f"Found {det['class']} with confidence {det['confidence']:.2f}")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Camera Not Found**: Check camera index (try 0, 1, 2...)
3. **Model Not Found**: Ensure model path is correct
4. **Import Errors**: Run setup.bat or install requirements manually

### Performance Tips

- Use GPU for training (CUDA)
- Reduce image size for faster inference
- Adjust confidence threshold based on needs
- Use appropriate YOLOv11 variant for your hardware

## Dataset Structure

```
HackByte_Dataset/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── classes.txt
└── yolo_params.yaml
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For issues and questions, please create an issue in the repository.
