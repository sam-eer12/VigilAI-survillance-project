import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['ultralytics', 'opencv-python', 'torch', 'pyyaml']
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def main():
    print("=== VigilAI Combined Model Training ===")
    print()
    print("This script will train a YOLOv11 medium model on all three datasets:")
    print("1. HackByte_Dataset (FireExtinguisher, ToolBox, OxygenTank)")
    print("2. FireSmokeNEWdataset (fire, smoke, other)")
    print("3. violence-detection-dataset (violence, non-violence)")
    print()
    
    # Check requirements
    print("Checking requirements...")
    check_requirements()
    
    # Start training
    print("\nStarting combined model training...")
    print("This may take several hours depending on your hardware.")
    print("Training will be done on CPU (safer for compatibility).")
    print()
    
    try:
        # Run the combined training
        from train_combined_model import main as train_main
        train_main()
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("\nTrying alternative method with vigilai.py...")
        
        # Alternative: use vigilai.py
        cmd = [
            sys.executable, 'vigilai.py',
            '--mode', 'train-combined',
            '--epochs', '50',
            '--batch', '4',
            '--img-size', '640',
            '--device', 'cpu'
        ]
        
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
