"""
Data organizer: reads metadata CSV and processes images into train/val/test splits.
Supports both CSDI and custom CSV formats with flexible severity-to-class mapping.
"""

import os
import shutil
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
from preprocess import standardize_eye_image


# Severity score mapping: adjust these ranges based on your dataset
SEVERITY_MAPPING_4CLASS = {
    0: "Healthy",      # [0, 1)
    1: "Mild",         # [1, 5)
    2: "Moderate",     # [5, 7)
    3: "Severe"        # [7, 10]
}

SEVERITY_MAPPING_5CLASS = {
    0: "Normal",       # [0, 1)
    1: "Acceptable",   # [1, 3)
    2: "Mild",         # [3, 5)
    3: "Moderate",     # [5, 7)
    4: "Severe"        # [7, 10]
}


def score_to_class(score, mapping=SEVERITY_MAPPING_4CLASS):
    """Map a continuous severity score to a class label."""
    score = float(score)
    if len(mapping) == 4:
        if score < 1:
            return 0
        elif score < 5:
            return 1
        elif score < 7:
            return 2
        else:
            return 3
    elif len(mapping) == 5:
        if score < 1:
            return 0
        elif score < 3:
            return 1
        elif score < 5:
            return 2
        elif score < 7:
            return 3
        else:
            return 4
    else:
        raise ValueError("Unsupported mapping size")


def load_metadata_csv(csv_path, image_col='id', score_col='score', diagnosis_col=None):
    """
    Load metadata from CSV file.
    Returns: list of (image_filename, score, diagnosis)
    """
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row[image_col]
            score = float(row[score_col])
            diagnosis = row.get(diagnosis_col, "") if diagnosis_col else ""
            data.append((filename, score, diagnosis))
    return data


def preprocess_and_organize(
    metadata_csv,
    source_dir,
    output_dir,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    num_classes=4,
    preprocess=True,
    target_size=(224, 224)
):
    """
    Read metadata CSV, preprocess images, and organize into train/val/test directories.
    
    Args:
        metadata_csv: Path to CSV with columns: id (filename), score, optional diagnosis
        source_dir: Directory containing original images
        output_dir: Root directory for train/val/test splits
        train_split: Fraction for training (default 0.7)
        val_split: Fraction for validation (default 0.15)
        test_split: Fraction for testing (default 0.15)
        num_classes: Number of severity classes (4 or 5)
        preprocess: Whether to apply standardize_eye_image (default True)
        target_size: Target image size for preprocessing
    """
    
    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        f"Splits must sum to 1.0, got {train_split + val_split + test_split}"
    
    mapping = SEVERITY_MAPPING_4CLASS if num_classes == 4 else SEVERITY_MAPPING_5CLASS
    
    # Load metadata
    print(f"Loading metadata from {metadata_csv}...")
    metadata = load_metadata_csv(metadata_csv)
    print(f"Loaded {len(metadata)} records.")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract images and labels
    images = []
    labels = []
    for filename, score, diagnosis in metadata:
        image_path = os.path.join(source_dir, filename)
        if not os.path.exists(image_path):
            print(f"Warning: image not found {image_path}, skipping.")
            continue
        label = score_to_class(score, mapping)
        images.append((image_path, filename, label))
        labels.append(label)
    
    print(f"Found {len(images)} valid images.")
    
    # Split into train/val/test
    train_idx, temp_idx = train_test_split(
        range(len(images)),
        train_size=train_split,
        stratify=labels,
        random_state=42
    )
    
    val_size = val_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        stratify=[labels[i] for i in temp_idx],
        random_state=42
    )
    
    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }
    
    # Create class directories and process images
    for split_name, split_indices in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"\nProcessing {split_name} split ({len(split_indices)} images)...")
        
        for idx in split_indices:
            src_path, filename, label = images[idx]
            class_name = mapping[label]
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Preprocess or copy
            if preprocess:
                try:
                    processed_img = standardize_eye_image(src_path, target_size=target_size)
                    if processed_img is not None:
                        out_path = os.path.join(class_dir, filename)
                        cv2.imwrite(out_path, processed_img)
                    else:
                        print(f"  Warning: preprocessing failed for {filename}")
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")
            else:
                dst_path = os.path.join(class_dir, filename)
                shutil.copy(src_path, dst_path)
        
        print(f"  {split_name.capitalize()} complete.")
    
    # Print summary
    print("\n=== Dataset Organization Complete ===")
    print(f"Output directory: {output_dir}")
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split_name)
        total = sum(len(os.listdir(os.path.join(split_dir, cls))) 
                   for cls in mapping.values() 
                   if os.path.isdir(os.path.join(split_dir, cls)))
        print(f"{split_name.capitalize()}: {total} images")
        for cls in mapping.values():
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                count = len(os.listdir(cls_dir))
                print(f"  {cls}: {count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Organize cataract dataset into train/val/test splits"
    )
    parser.add_argument("--csv", required=True, help="Path to metadata CSV")
    parser.add_argument("--source", required=True, help="Directory with original images")
    parser.add_argument("--output", default="./datasets", help="Output directory for splits")
    parser.add_argument("--train", type=float, default=0.7, help="Train split fraction")
    parser.add_argument("--val", type=float, default=0.15, help="Val split fraction")
    parser.add_argument("--test", type=float, default=0.15, help="Test split fraction")
    parser.add_argument("--classes", type=int, default=4, choices=[4, 5], help="Number of classes")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224], help="Target image size (H W)")
    
    args = parser.parse_args()
    
    preprocess_and_organize(
        metadata_csv=args.csv,
        source_dir=args.source,
        output_dir=args.output,
        train_split=args.train,
        val_split=args.val,
        test_split=args.test,
        num_classes=args.classes,
        preprocess=not args.no_preprocess,
        target_size=tuple(args.size)
    )
