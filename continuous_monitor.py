"""
Continuous eye health monitoring application.
Periodically tests images and tracks eye health changes over time.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import json

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from PIL import Image
import cv2
import numpy as np

from health_tracker import HealthTracker


class ContinuousHealthMonitor:
    """Monitor eye health continuously."""
    
    def __init__(
        self,
        model_path: str = "checkpoints/model_best.pt",
        tracking_file: str = "eye_health_history.json",
        device: str = None
    ):
        """
        Initialize monitor.
        
        Args:
            model_path: Path to trained model checkpoint
            tracking_file: Path to health history file
            device: torch device ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tracker = HealthTracker(tracking_file)
        self.model = None
        self.class_mapping = None
        
        # Load model once
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Reconstruct model
        num_classes = len(checkpoint.get('class_to_idx', {0: 'Healthy', 1: 'Severe'}))
        self.model = models.mobilenet_v2(pretrained=False)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get class mapping
        self.class_mapping = checkpoint.get('idx_to_class', {0: 'Healthy', 1: 'Severe'})
        print(f"✅ Model loaded. Classes: {list(self.class_mapping.values())}\n")
    
    def analyze_image(self, image_path: str, notes: str = "") -> dict:
        """
        Analyze a single image and track health.
        
        Args:
            image_path: Path to image file
            notes: Optional notes for this measurement
            
        Returns:
            Analysis result with health metrics
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        pred_class = self.class_mapping.get(pred_idx, f"Class {pred_idx}")
        
        # Extract probabilities
        healthy_prob = float(probs[0, 0].item())  # Assume index 0 is Healthy
        severe_prob = float(probs[0, 1].item()) if len(self.class_mapping) > 1 else 0
        
        # Record measurement
        self.tracker.record_measurement(
            image_path=image_path,
            healthy_prob=healthy_prob,
            severe_prob=severe_prob,
            predicted_class=pred_class,
            confidence=confidence,
            notes=notes
        )
        
        # Check for alerts
        health_score = healthy_prob * 100
        alert_triggered, alert_msg = self.tracker.check_alert_threshold(
            health_score,
            threshold_percent=10.0
        )
        
        result = {
            'image': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'predicted_class': pred_class,
            'confidence': round(confidence, 4),
            'probabilities': {
                'healthy': round(healthy_prob, 4),
                'severe': round(severe_prob, 4)
            },
            'health_score': round(health_score, 2),
            'alert_triggered': alert_triggered,
            'alert_message': alert_msg,
            'trend': self.tracker.get_health_trend(),
            'notes': notes
        }
        
        return result

    def analyze_pil_image(self, pil_image: Image.Image, image_label: str = "camera_frame", notes: str = "") -> dict:
        """
        Analyze a PIL Image directly (used for camera frames).
        """
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        img_tensor = transform(pil_image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

        pred_class = self.class_mapping.get(pred_idx, f"Class {pred_idx}")
        healthy_prob = float(probs[0, 0].item())
        severe_prob = float(probs[0, 1].item()) if probs.shape[1] > 1 else 0.0

        # Record measurement
        self.tracker.record_measurement(
            image_path=image_label,
            healthy_prob=healthy_prob,
            severe_prob=severe_prob,
            predicted_class=pred_class,
            confidence=confidence,
            notes=notes
        )

        health_score = healthy_prob * 100
        alert_triggered, alert_msg = self.tracker.check_alert_threshold(health_score, threshold_percent=10.0)

        result = {
            'image': image_label,
            'timestamp': datetime.now().isoformat(),
            'predicted_class': pred_class,
            'confidence': round(confidence, 4),
            'probabilities': {'healthy': round(healthy_prob, 4), 'severe': round(severe_prob, 4)},
            'health_score': round(health_score, 2),
            'alert_triggered': alert_triggered,
            'alert_message': alert_msg,
            'trend': self.tracker.get_health_trend(),
            'notes': notes
        }

        return result
    
    def monitor_directory(
        self,
        directory: str,
        interval_seconds: int = 300,
        max_iterations: int = None,
        recursive: bool = False
    ):
        """
        Monitor a directory for new eye images.
        
        Args:
            directory: Directory containing images
            interval_seconds: Check interval in seconds (default 5 min)
            max_iterations: Max iterations before stopping (None = infinite)
            recursive: Search subdirectories
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        print(f"🔍 Starting continuous monitoring on {directory}\n")
        print(f"Interval: {interval_seconds} seconds ({interval_seconds/60:.1f} minutes)")
        if max_iterations:
            print(f"Max iterations: {max_iterations}")
        print(f"Press Ctrl+C to stop\n")
        
        processed_files = set()
        iteration = 0
        
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] Scan #{iteration}")
                print("=" * 70)
                
                # Find image files
                glob_pattern = "**/*" if recursive else "*"
                image_files = list(directory.glob(glob_pattern))
                image_files = [
                    f for f in image_files
                    if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']
                ]
                
                if not image_files:
                    print("No images found.")
                else:
                    print(f"Found {len(image_files)} image(s)\n")
                    
                    for img_file in image_files:
                        file_key = str(img_file.absolute())
                        
                        if file_key not in processed_files:
                            print(f"📸 Processing: {img_file.name}")
                            
                            try:
                                result = self.analyze_image(
                                    str(img_file),
                                    notes=f"Continuous monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                )
                                
                                # Display result
                                self._display_result(result)
                                
                                # Alert if triggered
                                if result['alert_triggered']:
                                    print(f"\n🚨 {result['alert_message']}\n")
                                
                                processed_files.add(file_key)
                            
                            except Exception as e:
                                print(f"❌ Error processing {img_file.name}: {e}\n")
                
                # Wait for next iteration
                if max_iterations is None or iteration < max_iterations:
                    print(f"\nWaiting {interval_seconds} seconds until next scan...")
                    time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user.")
        
        finally:
            self._print_final_report()

    def monitor_camera(self,
                       camera_index: int = 0,
                       interval_seconds: int = 5,
                       max_iterations: Optional[int] = None,
                       save_frames: bool = False,
                       frame_save_dir: Optional[str] = None):
        """
        Monitor a camera feed, analyze frames at the given interval.
        Continuously captures eye images and tracks health changes.
        """
        print(f"🔍 Initializing camera access (index={camera_index})...")
        
        # Try to open camera with better error handling
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            # Try alternative camera indices
            for alt_index in range(5):
                if alt_index == camera_index:
                    continue
                print(f"Trying alternative camera index {alt_index}...")
                cap = cv2.VideoCapture(alt_index)
                if cap.isOpened():
                    camera_index = alt_index
                    print(f"✅ Successfully opened camera {alt_index}")
                    break
            else:
                raise RuntimeError(f"Cannot open any camera. Please check camera permissions and connections.")
        
        # Test camera by reading a frame
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            cap.release()
            raise RuntimeError("Camera opened but cannot read frames. Check camera functionality.")
        
        print(f"✅ Camera access granted. Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
        
        if save_frames and frame_save_dir:
            os.makedirs(frame_save_dir, exist_ok=True)
            print(f"📁 Frames will be saved to: {frame_save_dir}")

        print(f"🔴 Starting continuous eye health monitoring via camera")
        print(f"Interval between analyses: {interval_seconds} seconds")
        print("Position your eye in front of the camera for monitoring")
        print("Press Ctrl+C to stop monitoring\n")

        iteration = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Camera capture #{iteration}")
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    print(f"⚠️  Failed to capture frame (failure {consecutive_failures}/{max_consecutive_failures})")
                    if consecutive_failures >= max_consecutive_failures:
                        print("❌ Too many consecutive capture failures. Stopping monitoring.")
                        break
                    time.sleep(interval_seconds)
                    continue
                
                consecutive_failures = 0  # Reset on success
                
                # Convert BGR to RGB and to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                label = f"camera_{camera_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                if save_frames and frame_save_dir:
                    save_path = os.path.join(frame_save_dir, label)
                    cv2.imwrite(save_path, frame)
                    image_label = save_path
                else:
                    image_label = label

                print(f"📸 Analyzing eye image from camera...")
                result = self.analyze_pil_image(pil_img, image_label=image_label, notes="continuous_camera_monitoring")
                self._display_result(result)
                if result['alert_triggered']:
                    print(f"\n🚨 {result['alert_message']}\n")
                
                # Wait for next capture
                if max_iterations is None or iteration < max_iterations:
                    time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n⏹️  Camera monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Error during camera monitoring: {e}")
        finally:
            cap.release()
            print("📷 Camera released")
            self._print_final_report()
            self._print_final_report()
    
    def _display_result(self, result: dict):
        """Display analysis result."""
        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Health Score: {result['health_score']:.1f}/100")
        print(f"  Healthy Probability: {result['probabilities']['healthy']:.1%}")
        print(f"  Severe Probability: {result['probabilities']['severe']:.1%}")
        
        trend = result['trend']
        if 'latest_score' in trend:
            print(f"\n  📊 Trend:")
            print(f"    Direction: {trend['trend_direction']}")
            print(f"    Change: {trend['change']:+.2f} ({trend['percent_change']:+.1f}%)")
            print(f"    Severity: {trend['trend_severity']}")
    
    def _print_final_report(self):
        """Print final session report."""
        stats = self.tracker.get_statistics()
        
        if stats.get('status') == 'no_data':
            print("\nNo measurements recorded.")
            return
        
        print("\n" + "=" * 70)
        print("📊 MONITORING SESSION REPORT")
        print("=" * 70)
        
        print("\n📈 Statistics:")
        print(f"  Total measurements: {stats['total_measurements']}")
        print(f"  First: {stats['first_measurement']}")
        print(f"  Latest: {stats['latest_measurement']}")
        
        hs = stats['health_score']
        print(f"\n💚 Health Score:")
        print(f"  Current: {hs['current']:.1f}/100")
        print(f"  Average: {hs['mean']:.1f}/100")
        print(f"  Range: {hs['min']:.1f} - {hs['max']:.1f}")
        
        pd = stats['prediction_distribution']
        print(f"\n🔍 Predictions:")
        print(f"  Healthy: {pd['healthy']} ({pd['healthy_percent']:.1f}%)")
        print(f"  Severe: {pd['severe']} ({pd['severe_percent']:.1f}%)")
        
        print(f"\n✅ Report saved to: {self.tracker.export_report()}")
        print("=" * 70 + "\n")
    
    def get_latest_result(self) -> Optional[dict]:
        """Get the latest measurement."""
        history = self.tracker.get_all_measurements()
        if not history:
            return None
        
        latest = history[-1]
        return {
            'timestamp': latest['timestamp'],
            'predicted_class': latest['predicted_class'],
            'confidence': latest['confidence'],
            'health_score': latest['health_score'],
            'healthy_prob': latest['healthy_prob'],
            'severe_prob': latest['severe_prob']
        }


def main():
    parser = argparse.ArgumentParser(
        description="Continuous eye health monitoring system"
    )
    parser.add_argument(
        "mode",
        choices=["single", "directory", "watch", "camera"],
        help="Operating mode"
    )
    parser.add_argument(
        "--image",
        help="Image path (for single mode)"
    )
    parser.add_argument(
        "--directory",
        default="./test_images",
        help="Directory to monitor (for directory/watch mode)"
    )
    parser.add_argument(
        "--model",
        default="checkpoints/model_best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--tracking-file",
        default="eye_health_history.json",
        help="Path to health history file"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (for watch mode)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Max iterations (for watch mode)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to use (auto-detect if not specified)"
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Notes to add to measurement"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index (for camera mode)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save captured frames to disk (camera mode)"
    )
    parser.add_argument(
        "--frame-save-dir",
        default="./camera_frames",
        help="Directory to save frames if --save-frames is used"
    )
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = ContinuousHealthMonitor(
        model_path=args.model,
        tracking_file=args.tracking_file,
        device=args.device
    )
    
    # Execute mode
    if args.mode == "single":
        if not args.image:
            print("❌ --image required for single mode")
            return 1
        
        print(f"📸 Analyzing: {args.image}\n")
        result = monitor.analyze_image(args.image, notes=args.notes)
        monitor._display_result(result)
        
        if result['alert_triggered']:
            print(f"\n🚨 Alert:\n{result['alert_message']}")
        
        # Save result
        with open("latest_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Result saved to latest_result.json")
    
    elif args.mode == "directory":
        # Single scan of directory
        monitor.monitor_directory(
            directory=args.directory,
            interval_seconds=0,  # No wait
            max_iterations=1,
            recursive=True
        )
    
    elif args.mode == "watch":
        # Continuous monitoring
        monitor.monitor_directory(
            directory=args.directory,
            interval_seconds=args.interval,
            max_iterations=args.iterations,
            recursive=True
        )
    elif args.mode == "camera":
        monitor.monitor_camera(
            camera_index=args.camera_index,
            interval_seconds=args.interval if args.interval else 5,
            max_iterations=args.iterations,
            save_frames=args.save_frames,
            frame_save_dir=args.frame_save_dir
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
