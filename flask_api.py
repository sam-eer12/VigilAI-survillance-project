from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
import time
import json
import os
from datetime import datetime
import cv2
import logging
from vigilai import VigilAI

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global variables
vigilai_instance = None
detection_thread = None
training_thread = None
evaluation_thread = None
system_logs = []
system_status = "Idle"
detection_running = False
camera_feed = None
latest_frame = None  # JPEG-encoded latest annotated frame bytes
video_capture = None

def add_log(message, level="INFO"):
    """Add log entry with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message
    }
    system_logs.append(log_entry)
    # Keep only last 100 logs
    if len(system_logs) > 100:
        system_logs.pop(0)
    print(f"[{timestamp}] {level}: {message}")

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status"""
    return jsonify({
        "status": system_status,
        "detection_running": detection_running,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get recent system logs"""
    limit = request.args.get('limit', 50, type=int)
    return jsonify({
        "logs": system_logs[-limit:],
        "total": len(system_logs)
    })

@app.route('/api/train', methods=['POST'])
def start_training():
    """Start model training"""
    global training_thread, system_status, vigilai_instance
    
    if system_status != "Idle":
        return jsonify({"error": "System is busy"}), 400
    
    data = request.get_json()
    epochs = data.get('epochs', 20)
    batch_size = data.get('batch_size', 16)
    img_size = data.get('img_size', 416)
    model_size = data.get('model_size', 'm')
    training_type = data.get('training_type', 'standard')  # 'standard' or 'combined'
    
    def train_model():
        global system_status
        try:
            system_status = "Training"
            
            if training_type == 'combined':
                add_log(f"Starting COMBINED training: epochs={epochs}, batch={batch_size}, img_size={img_size}, model_size={model_size}")
                add_log("Training on all three datasets: HackByte + FireSmoke + Violence Detection")
                
                # Check if datasets are available
                vigilai_instance = VigilAI(model_size=model_size)
                if not vigilai_instance.check_datasets_available():
                    add_log("Warning: Some datasets missing, but continuing with available data", "WARNING")
                
                # Use the combined training approach
                from train_combined_model import CombinedDatasetTrainer
                
                trainer = CombinedDatasetTrainer()
                add_log("Setting up combined dataset...")
                trainer.setup_combined_dataset()
                
                add_log("Starting combined model training...")
                model_path = trainer.train_combined_model(
                    epochs=epochs,
                    img_size=img_size,
                    batch_size=batch_size,
                    device='cpu'
                )
                
                if model_path:
                    add_log("Combined training completed successfully!")
                    add_log(f"Model saved at: {model_path}")
                    add_log(f"Classes: {', '.join(trainer.combined_classes)}")
                else:
                    add_log("Combined training failed!", "ERROR")
                    
            else:
                # Standard training on HackByte dataset only
                add_log(f"Starting STANDARD training: epochs={epochs}, batch={batch_size}, img_size={img_size}, model_size={model_size}")
                add_log("Training on HackByte dataset only")
                
                vigilai_instance = VigilAI(model_size=model_size)
                vigilai_instance.extract_dataset()
                results = vigilai_instance.train_yolov11(
                    epochs=epochs,
                    img_size=img_size,
                    batch_size=batch_size
                )
                
                if results:
                    add_log("Standard training completed successfully!")
                else:
                    add_log("Standard training failed!", "ERROR")
                
        except Exception as e:
            add_log(f"Training error: {str(e)}", "ERROR")
        finally:
            system_status = "Idle"
    
    training_thread = threading.Thread(target=train_model)
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({"message": "Training started", "parameters": data})

@app.route('/api/detect/start', methods=['POST'])
def start_detection():
    """Start live detection"""
    global detection_thread, system_status, detection_running, vigilai_instance
    
    if detection_running:
        return jsonify({"error": "Detection already running"}), 400
    
    data = request.get_json()
    camera_index = data.get('camera_index', 0)
    conf_threshold = data.get('conf_threshold', 0.5)
    model_size = data.get('model_size', 'm')
    model_path = data.get('model_path', None)
    
    def run_detection():
        global detection_running, system_status, latest_frame, video_capture
        try:
            detection_running = True
            system_status = "Detecting"
            add_log(f"Starting detection: camera={camera_index}, conf={conf_threshold}, model_size={model_size}")
            
            vigilai_instance = VigilAI(model_size=model_size)
            if model_path:
                vigilai_instance.model_path = model_path
            
            # Find latest model if no path specified
            if not model_path:
                from pathlib import Path
                runs_dir = Path('runs/train')
                if runs_dir.exists():
                    latest_run = max(runs_dir.glob('vigilai_yolov11*'), 
                                   key=lambda x: x.stat().st_mtime, default=None)
                    if latest_run and (latest_run / 'weights' / 'best.pt').exists():
                        vigilai_instance.model_path = str(latest_run / 'weights' / 'best.pt')
            
            if vigilai_instance.load_model():
                # Initialize camera
                video_capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if not video_capture.isOpened():
                    add_log(f"Error: Could not open camera {camera_index}", "ERROR")
                    return

                add_log("Camera opened successfully. Streaming frames...")

                # Try to set target FPS to 60
                try:
                    video_capture.set(cv2.CAP_PROP_FPS, 60)
                except Exception:
                    pass

                target_interval = 1.0 / 60.0

                while detection_running:
                    loop_start = time.perf_counter()
                    ret, frame = video_capture.read()
                    if not ret:
                        add_log("Error: Could not read frame from camera", "ERROR")
                        break

                    # Run detection and annotate
                    annotated_frame, _ = vigilai_instance.detect_objects(frame, conf_threshold)

                    # Encode as JPEG
                    ok, buffer = cv2.imencode('.jpg', annotated_frame)
                    if ok:
                        latest_frame = buffer.tobytes()
                    else:
                        add_log("Warning: Failed to encode frame", "WARNING")

                    # Pace loop to target ~60 FPS
                    elapsed = time.perf_counter() - loop_start
                    remaining = target_interval - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
            else:
                add_log("Failed to load model", "ERROR")
                
        except Exception as e:
            add_log(f"Detection error: {str(e)}", "ERROR")
        finally:
            detection_running = False
            system_status = "Idle"
            add_log("Detection stopped")
            try:
                if video_capture is not None:
                    video_capture.release()
                video_capture = None
            except Exception:
                pass
            latest_frame = None
    
    detection_thread = threading.Thread(target=run_detection)
    detection_thread.daemon = True
    detection_thread.start()
    
    return jsonify({"message": "Detection started", "parameters": data})

@app.route('/api/detect/stop', methods=['POST'])
def stop_detection():
    """Stop live detection"""
    global detection_running, system_status
    
    if not detection_running:
        return jsonify({"error": "Detection not running"}), 400
    
    detection_running = False
    system_status = "Idle"
    add_log("Detection stop requested")
    
    return jsonify({"message": "Detection stop requested"})

@app.route('/api/stream', methods=['GET'])
def stream_video():
    """Stream the latest annotated frames as MJPEG."""
    def generate():
        global latest_frame
        boundary = b'--frame\r\n'
        while True:
            if latest_frame is not None:
                frame = latest_frame
                headers = [
                    b'Content-Type: image/jpeg\r\n',
                    f'Content-Length: {len(frame)}\r\n'.encode('utf-8'),
                    b'\r\n'
                ]
                yield boundary + b''.join(headers) + frame + b'\r\n'
            else:
                time.sleep(0.05)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/screenshot', methods=['POST'])
def save_screenshot():
    """Save the current frame as a screenshot"""
    global latest_frame
    
    if not detection_running:
        return jsonify({"error": "Detection not running"}), 400
        
    if latest_frame is None:
        return jsonify({"error": "No frame available"}), 400
    
    try:
        # Create screenshots directory if it doesn't exist
        import os
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vigilai_screenshot_{timestamp}.jpg"
        filepath = os.path.join(screenshots_dir, filename)
        
        # Save the frame
        with open(filepath, 'wb') as f:
            f.write(latest_frame)
        
        add_log(f"Screenshot saved: {filename}")
        
        return jsonify({
            "message": "Screenshot saved successfully",
            "filename": filename,
            "filepath": filepath,
            "timestamp": timestamp
        })
        
    except Exception as e:
        add_log(f"Screenshot error: {str(e)}", "ERROR")
        return jsonify({"error": f"Failed to save screenshot: {str(e)}"}), 500

@app.route('/api/evaluate', methods=['POST'])
def start_evaluation():
    """Start model evaluation"""
    global evaluation_thread, system_status, vigilai_instance
    
    if system_status != "Idle":
        return jsonify({"error": "System is busy"}), 400
    
    data = request.get_json()
    model_path = data.get('model_path', None)
    evaluation_type = data.get('evaluation_type', 'standard')  # 'standard' or 'combined'
    
    def run_evaluation():
        global system_status
        try:
            system_status = "Evaluating"
            add_log(f"Starting {evaluation_type} evaluation with model: {model_path}")
            
            vigilai_instance = VigilAI(model_path=model_path)
            
            if evaluation_type == 'combined':
                # For combined models, evaluate on combined dataset
                add_log("Evaluating on combined dataset...")
                results = vigilai_instance.evaluate_combined_model()
            else:
                # Standard evaluation on HackByte dataset
                add_log("Evaluating on HackByte dataset...")
                results = vigilai_instance.evaluate_model()
            
            if results:
                add_log("Evaluation completed successfully!")
                add_log(f"Results: {results}")
            else:
                add_log("Evaluation failed!", "ERROR")
                
        except Exception as e:
            add_log(f"Evaluation error: {str(e)}", "ERROR")
        finally:
            system_status = "Idle"
    
    evaluation_thread = threading.Thread(target=run_evaluation)
    evaluation_thread.daemon = True
    evaluation_thread.start()
    
    return jsonify({"message": "Evaluation started", "model_path": model_path, "evaluation_type": evaluation_type})

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available trained models"""
    try:
        from pathlib import Path
        models = []
        runs_dir = Path('runs/train')
        
        if runs_dir.exists():
            for run_dir in runs_dir.glob('vigilai_yolov11*'):
                weights_dir = run_dir / 'weights'
                if weights_dir.exists():
                    best_model = weights_dir / 'best.pt'
                    if best_model.exists():
                        models.append({
                            "name": f"{run_dir.name} (best)",
                            "path": str(best_model),
                            "modified": best_model.stat().st_mtime
                        })
        
        models.sort(key=lambda x: x['modified'], reverse=True)
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/datasets', methods=['GET'])
def get_dataset_status():
    """Get status of available datasets"""
    try:
        from pathlib import Path
        datasets = {
            "HackByte_Dataset": Path("HackByte_Dataset").exists(),
            "FireSmokeNEWdataset.v1i.yolov9": Path("FireSmokeNEWdataset.v1i.yolov9").exists(),
            "violence-detection-dataset": Path("violence-detection-dataset").exists(),
            "Hackathon_Dataset.zip": Path("Hackathon_Dataset.zip").exists()
        }
        
        available_count = sum(datasets.values())
        total_count = len(datasets)
        
        return jsonify({
            "datasets": datasets,
            "available_count": available_count,
            "total_count": total_count,
            "can_train_combined": available_count >= 2  # Need at least 2 datasets for combined training
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-info/<path:model_path>', methods=['GET'])
def get_model_info(model_path):
    """Get detailed information about a specific model"""
    try:
        from pathlib import Path
        import os
        
        # Decode the model path
        model_path = model_path.replace('_', '/')
        
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found"}), 404
        
        # Get model file size
        model_size = os.path.getsize(model_path)
        model_size_mb = model_size / (1024 * 1024)
        
        # Determine model type based on path
        if 'combined' in model_path:
            model_type = "Combined (Multi-Dataset)"
            classes = ['FireExtinguisher', 'ToolBox', 'OxygenTank', 'fire', 'smoke', 'other', 'violence', 'non-violence']
        else:
            model_type = "Standard (HackByte)"
            classes = ['FireExtinguisher', 'ToolBox', 'OxygenTank']
        
        # Get model modification time
        modified_time = os.path.getmtime(model_path)
        
        # Try to get model parameters (this is an estimate)
        if 'yolov11n' in model_path or 'nano' in model_path:
            parameters = "2.6M"
            inference_speed = "1.2ms CPU"
        elif 'yolov11s' in model_path or 'small' in model_path:
            parameters = "9.4M"
            inference_speed = "2.1ms CPU"
        elif 'yolov11m' in model_path or 'medium' in model_path:
            parameters = "20.1M"
            inference_speed = "4.5ms CPU"
        elif 'yolov11l' in model_path or 'large' in model_path:
            parameters = "25.3M"
            inference_speed = "6.8ms CPU"
        elif 'yolov11x' in model_path or 'xlarge' in model_path:
            parameters = "48.1M"
            inference_speed = "12.3ms CPU"
        else:
            parameters = "Unknown"
            inference_speed = "Unknown"
        
        return jsonify({
            "model_path": model_path,
            "model_type": model_type,
            "model_size_mb": round(model_size_mb, 2),
            "parameters": parameters,
            "inference_speed": inference_speed,
            "classes": classes,
            "modified_time": modified_time,
            "can_evaluate_combined": 'combined' in model_path
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluation-results/<path:model_path>', methods=['GET'])
def get_evaluation_results(model_path):
    """Get evaluation results for a specific model"""
    try:
        from pathlib import Path
        import json
        import pandas as pd
        
        # Decode the model path
        model_path = model_path.replace('_', '/')
        
        # Look for evaluation results in the model's directory
        model_dir = Path(model_path).parent.parent
        results_file = model_dir / 'results.json'
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            return jsonify(results)
        else:
            return jsonify({
                "error": "No evaluation results found",
                "message": "Run evaluation first to generate results"
            }), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/training-results/<path:model_path>', methods=['GET'])
def get_training_results(model_path):
    """Get training results from results.csv for a specific model"""
    try:
        from pathlib import Path
        import pandas as pd
        
        # Decode the model path
        model_path = model_path.replace('_', '/')
        
        # Look for results.csv in the model's directory
        model_dir = Path(model_path).parent.parent
        results_csv = model_dir / 'results.csv'
        
        if not results_csv.exists():
            return jsonify({
                "error": "No training results found",
                "message": "Training results CSV file not found"
            }), 404
        
        # Read the CSV file
        df = pd.read_csv(results_csv)
        
        if df.empty:
            return jsonify({
                "error": "No training data found",
                "message": "Training results CSV is empty"
            }), 404
        
        # Get the last row (final epoch results)
        final_results = df.iloc[-1]
        
        # Extract key metrics
        training_results = {
            "total_epochs": len(df),
            "final_epoch": int(final_results['epoch']),
            "training_time": final_results['time'],
            
            # Training losses
            "train_box_loss": float(final_results['train/box_loss']),
            "train_cls_loss": float(final_results['train/cls_loss']),
            "train_dfl_loss": float(final_results['train/dfl_loss']),
            
            # Validation losses
            "val_box_loss": float(final_results['val/box_loss']),
            "val_cls_loss": float(final_results['val/cls_loss']),
            "val_dfl_loss": float(final_results['val/dfl_loss']),
            
            # Metrics
            "precision": float(final_results['metrics/precision(B)']),
            "recall": float(final_results['metrics/recall(B)']),
            "mAP50": float(final_results['metrics/mAP50(B)']),
            "mAP50_95": float(final_results['metrics/mAP50-95(B)']),
            
            # Learning rates
            "lr_pg0": float(final_results['lr/pg0']),
            "lr_pg1": float(final_results['lr/pg1']),
            "lr_pg2": float(final_results['lr/pg2']),
            
            # Historical data for charts
            "epoch_data": df.to_dict('records')
        }
        
        return jsonify(training_results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    add_log("VigilAI Flask API Server starting...")
    add_log("System initialized and ready for operations")
    add_log("API endpoints available: /api/status, /api/logs, /api/train, /api/detect/*, /api/evaluate, /api/models")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
