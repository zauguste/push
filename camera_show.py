"""
Simple camera monitoring demo that displays the webcam feed with
prediction overlays using the trained cataract model.

This script avoids the locked `continuous_monitor.py` file by reusing
`predict.py` utilities, so it's safe to run even when the original
module is inaccessible.

Usage:
    python camera_show.py [--device cpu|cuda]

Press 'q' in the window to quit early.
"""

import argparse
import time

import cv2
from PIL import Image
import torch
from torchvision.transforms import transforms

from predict import load_checkpoint
from health_tracker import HealthTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tracking-file", default="eye_health_history.json")
    args = parser.parse_args()

    # load model
    print("Loading model...")
    model, class_mapping = load_checkpoint("checkpoints/model_best.pt", device=args.device)
    print(f"Model ready. Classes: {list(class_mapping.values())}")

    tracker = HealthTracker(args.tracking_file)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera; check permissions/connection")

    print("Press 'q' in the window to stop")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # convert to PIL and preprocess
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

            tracker.record_measurement(
                image_path="camera_feed",
                healthy_prob=healthy_prob,
                severe_prob=severe_prob,
                predicted_class=pred_class,
                confidence=confidence,
                notes="live_camera"
            )

            # overlay
            text = f"{pred_class} {confidence:.2%}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.putText(frame, f"Score:{health_score:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            cv2.imshow("Eye Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Session ended")
        stats = tracker.get_statistics()
        if stats.get('status') != 'no_data':
            print(f"Total measurements: {stats['total_measurements']}")


if __name__ == "__main__":
    main()
