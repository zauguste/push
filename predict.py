"""
Inference script: Load trained model and predict on single images or batches.
Integrates with health tracking for continuous monitoring.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json
import argparse

try:
    from health_tracker import HealthTracker
    HEALTH_TRACKING_AVAILABLE = True
except ImportError:
    HEALTH_TRACKING_AVAILABLE = False


def load_checkpoint(checkpoint_path, num_classes=2, device='cpu'):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build model
    model = models.mobilenet_v2(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get class mapping if available
    class_mapping = checkpoint.get('idx_to_class', {0: 'Healthy', 1: 'Severe'})
    
    return model, class_mapping


def predict_image(
    image_path,
    model,
    class_mapping,
    device='cpu',
    return_probs=True
):
    """
    Predict on a single image.
    
    Args:
        image_path: Path to image file
        model: PyTorch model in eval mode
        class_mapping: Dict mapping index to class name
        device: torch device
        return_probs: If True, return probabilities
        
    Returns:
        (predicted_class, confidence, probabilities_dict)
    """
    
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
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    pred_class = class_mapping.get(pred_idx, f"Class {pred_idx}")
    
    result = {
        'image': str(image_path),
        'predicted_class': pred_class,
        'confidence': f"{confidence:.4f}",
    }
    
    if return_probs:
        result['probabilities'] = {
            class_mapping.get(i, f"Class {i}"): float(probs[0, i].item())
            for i in range(probs.shape[1])
        }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict on cataract images")
    parser.add_argument("--image", required=True, help="Path to image file (or directory for batch)")
    parser.add_argument("--model", default="checkpoints/model_best.pt", help="Path to model checkpoint")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--output", default=None, help="JSON output file (optional)")
    parser.add_argument("--track-health", action="store_true", help="Track health history")
    parser.add_argument("--tracking-file", default="eye_health_history.json", help="Health history file")
    parser.add_argument("--notes", default="", help="Notes for this measurement")
    parser.add_argument("--camera", action="store_true", help="Run live camera feed (overrides --image)")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, class_mapping = load_checkpoint(args.model, device=args.device)
    print(f"✅ Model loaded. Classes: {list(class_mapping.values())}\n")
    
    # Initialize health tracker if requested
    tracker = None
    if args.track_health and HEALTH_TRACKING_AVAILABLE:
        tracker = HealthTracker(args.tracking_file)
        print(f"📊 Health tracking enabled\n")
    
    # if camera mode was requested, ignore image path and open webcam
    if args.camera:
        if not HEALTH_TRACKING_AVAILABLE and not tracker:
            tracker = HealthTracker(args.tracking_file)
        print("🔍 Starting camera mode. Press 'q' window to quit.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_tensor = transform(pil).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    logits = model(img_tensor)
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = probs.argmax(dim=1).item()
                    confidence = probs[0, pred_idx].item()
                pred_class = class_mapping.get(pred_idx, f"Class {pred_idx}")
                healthy_prob = float(probs[0, 0].item())
                severe_prob = float(probs[0, 1].item()) if probs.shape[1] > 1 else 0.0
                health_score = healthy_prob * 100
                if tracker:
                    tracker.record_measurement(
                        image_path="camera",
                        healthy_prob=healthy_prob,
                        severe_prob=severe_prob,
                        predicted_class=pred_class,
                        confidence=confidence,
                        notes="camera_mode"
                    )
                text = f"{pred_class} {confidence:.2%}"
                cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Score:{health_score:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.imshow('Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print('Camera session ended')
            if tracker:
                stats = tracker.get_statistics()
                print(f"Total measurements: {stats.get('total_measurements')}")
        sys.exit(0)
    
    # Predict
    image_path = Path(args.image)
    
    if image_path.is_file() and image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        # Single image
        print(f"Predicting on {image_path}...")
        result = predict_image(image_path, model, class_mapping, device=args.device)
        
        print(f"\nResult:")
        print(f"  Image: {result['image']}")
        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"    {cls}: {prob:.4f}")
        
        # Track health if enabled
        if tracker:
            # Extract health and severe probabilities
            healthy_prob = result['probabilities'].get('Healthy', 
                                                       result['probabilities'].get('0', 0.5))
            severe_prob = result['probabilities'].get('Severe',
                                                      result['probabilities'].get('1', 0.5))
            
            # Record measurement
            measurement = tracker.record_measurement(
                image_path=str(image_path),
                healthy_prob=healthy_prob,
                severe_prob=severe_prob,
                predicted_class=result['predicted_class'],
                confidence=float(result['confidence']),
                notes=args.notes or f"Single image prediction - {image_path.name}"
            )
            
            print(f"\n📊 Health Tracking:")
            print(f"  Health Score: {measurement['health_score']:.1f}/100")
            print(f"  Healthy Probability: {measurement['healthy_prob']:.1%}")
            print(f"  Severe Probability: {measurement['severe_prob']:.1%}")
            
            # Check for alerts
            alert, msg = tracker.check_alert_threshold(
                measurement['health_score'],
                threshold_percent=10.0
            )
            
            if alert:
                print(f"\n🚨 ALERT:\n{msg}")
            
            # Show trend if available
            trend = tracker.get_health_trend()
            if 'latest_score' in trend:
                print(f"\n📈 Trend:")
                print(f"  Latest: {trend['latest_score']:.1f}")
                print(f"  Change: {trend['change']:+.2f} ({trend['percent_change']:+.1f}%)")
                print(f"  Direction: {trend['trend_direction']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✅ Results saved to {args.output}")
    
    elif image_path.is_dir():
        # Batch prediction
        results = []
        image_files = list(image_path.glob('*.png')) + list(image_path.glob('*.jpg')) + list(image_path.glob('*.jpeg'))
        
        print(f"Batch predicting on {len(image_files)} images...\n")
        
        for i, img_file in enumerate(image_files, 1):
            result = predict_image(img_file, model, class_mapping, device=args.device)
            results.append(result)
            print(f"[{i}/{len(image_files)}] {img_file.name}: {result['predicted_class']} ({result['confidence']})")
            
            # Track if enabled
            if tracker:
                healthy_prob = result['probabilities'].get('Healthy', 0.5)
                severe_prob = result['probabilities'].get('Severe', 0.5)
                tracker.record_measurement(
                    image_path=str(img_file),
                    healthy_prob=healthy_prob,
                    severe_prob=severe_prob,
                    predicted_class=result['predicted_class'],
                    confidence=float(result['confidence']),
                    notes=f"Batch prediction - {img_file.name}"
                )
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to {args.output}")
        
        # Print batch statistics
        if tracker:
            stats = tracker.get_statistics()
            print(f"\n📊 Batch Health Statistics:")
            print(f"  Total predictions: {len(results)}")
            if 'prediction_distribution' in stats:
                pd = stats['prediction_distribution']
                print(f"  Healthy: {pd.get('healthy', 0)} ({pd.get('healthy_percent', 0):.1f}%)")
                print(f"  Severe: {pd.get('severe', 0)} ({pd.get('severe_percent', 0):.1f}%)")
    
    else:
        print(f"❌ {image_path} is not a file or directory")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
