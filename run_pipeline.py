"""
Master script to run the complete cataract detection pipeline:
1. Verify dependencies
2. Train model with MLflow tracking
3. Display results and start UI
"""

import subprocess
import sys
import os
import time
from pathlib import Path


def run_command(cmd, description, shell=False):
    """Run a command and report results."""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            print(f"\n✅ {description} — SUCCESS")
            return True
        else:
            print(f"\n❌ {description} — FAILED (exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"\n❌ {description} — ERROR: {e}")
        return False


def main():
    """Run the full pipeline."""
    
    print("\n" + "="*70)
    print("🚀 CATARACT DETECTION PIPELINE — FULL RUN")
    print("="*70)
    
    # Check datasets exist
    print("\n📊 Checking datasets...")
    datasets_path = Path("datasets")
    if not datasets_path.exists():
        print("❌ datasets/ directory not found!")
        return False
    
    train_count = len(list(Path("datasets/train").glob("*/*")))
    test_count = len(list(Path("datasets/test").glob("*/*")))
    original_count = len(list(Path("datasets/original_images").glob("*")))
    
    print(f"  ✅ Original images: {original_count}")
    print(f"  ✅ Training images: {train_count}")
    print(f"  ✅ Test images: {test_count}")
    
    if train_count == 0 or test_count == 0:
        print("\n❌ No training or test data found! Run import_datasets.py first.")
        return False
    
    # Train model
    print("\n" + "="*70)
    print("🤖 TRAINING MODEL")
    print("="*70)
    
    train_cmd = [
        sys.executable,
        "train.py",
        "--train-dir", "datasets/train",
        "--test-dir", "datasets/test",
        "--epochs", "10",
        "--batch-size", "32",
        "--lr", "1e-4",
        "--experiment", "cataract_detection",
        "--run-name", f"full_run_{int(time.time())}"
    ]
    
    success = run_command(train_cmd, "Training Model")
    
    if not success:
        print("\n❌ Training failed!")
        return False
    
    # Show MLflow results
    print("\n" + "="*70)
    print("📈 VIEWING BEST RUN")
    print("="*70)
    
    best_cmd = [
        sys.executable,
        "mlflow_utils.py",
        "best",
        "--experiment", "cataract_detection",
        "--metric", "test_acc"
    ]
    
    run_command(best_cmd, "Getting Best Run Metrics")
    
    # List all experiments
    print("\n" + "="*70)
    print("📊 ALL EXPERIMENTS")
    print("="*70)
    
    list_cmd = [sys.executable, "mlflow_utils.py", "list"]
    run_command(list_cmd, "Listing All Experiments")
    
    # Summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print("""
📁 Project Structure:
  datasets/
    ├── original_images/        [2,640 nuclear cataract images]
    ├── train/                  [491 Healthy/Severe images]
    └── test/                   [121 Healthy/Severe images]
  
📊 Checkpoints:
    checkpoints/
    ├── model_best.pt           [Best validation accuracy]
    ├── model_final.pt          [Final epoch model]
    └── history.json            [Training curves]
    
🔍 MLflow Tracking:
    mlruns/
    └── 1/cataract_detection/   [All experiment runs]
    
🚀 Next Steps:

  1. START MLFLOW DASHBOARD:
     python mlflow_utils.py ui --port 5000
     → Open http://localhost:5000 in browser
     
  2. TRAIN AGAIN WITH DIFFERENT PARAMS:
     python train.py --epochs 20 --batch-size 64 --lr 5e-4 --run-name "tuning_v1"
     
  3. PREPROCESS NUCLEAR DATASET:
     python organize_dataset.py --csv CSDI_annotations.csv --source datasets/original_images --classes 4
     
  4. CONVERT TO iOS (CoreML):
     python -c "from src.model import convert_to_coreml; convert_to_coreml(...)"
     
  5. RUN INFERENCE:
     python predict.py --image path/to/image.jpg --model checkpoints/model_best.pt
""")
    
    print("="*70)
    return True


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    success = main()
    sys.exit(0 if success else 1)
