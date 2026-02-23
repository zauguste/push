"""
Import and organize two cataract datasets:
1. Nuclear Cataract Database - raw patient images (DER/IZQ = Right/Left eyes)
2. Processed Images - pre-split train/test with cataract/normal labels
"""

import os
import shutil
from pathlib import Path


def import_nuclear_dataset(
    source_dir,
    output_original_images_dir,
    flatten=True
):
    """
    Import Nuclear Cataract Database patient images.
    
    Args:
        source_dir: Path containing 01, 02, ..., 115 patient folders
        output_original_images_dir: Where to copy images
        flatten: If True, copy all images to one dir. If False, keep patient hierarchy.
    """
    os.makedirs(output_original_images_dir, exist_ok=True)
    
    patient_dirs = sorted([d for d in Path(source_dir).iterdir() if d.is_dir() and d.name.isdigit()])
    
    total = 0
    for i, patient_dir in enumerate(patient_dirs):
        print(f"  [{i+1}/{len(patient_dirs)}] Processing {patient_dir.name}...")
        for eye_dir in ['DER', 'IZQ', 'Der', 'Izq']:  # Handle case variations
            eye_path = patient_dir / eye_dir
            if eye_path.exists():
                for img_file in eye_path.glob('IM*.JPG'):
                    if flatten:
                        # Flatten: copy to single dir with unique name
                        dst_name = f"nuclear_{patient_dir.name}_{eye_dir}_{img_file.name}"
                        dst_path = os.path.join(output_original_images_dir, dst_name)
                    else:
                        # Hierarchy: preserve patient structure
                        patient_output = os.path.join(output_original_images_dir, f"patient_{patient_dir.name}", eye_dir)
                        os.makedirs(patient_output, exist_ok=True)
                        dst_path = os.path.join(patient_output, img_file.name)
                    
                    shutil.copy2(img_file, dst_path)
                    total += 1
    
    print(f"Imported {total} images from Nuclear dataset to {output_original_images_dir}")
    return total


def import_processed_dataset(
    source_dir,
    output_train_dir,
    output_test_dir,
    label_mapping={
        'normal': 'Healthy',
        'cataract': 'Severe'
    }
):
    """
    Import pre-processed dataset that's already split into train/test and labeled.
    
    Args:
        source_dir: Path containing train/ and test/ subdirs
        output_train_dir: Root output dir for training split
        output_test_dir: Root output dir for test split
        label_mapping: Map source labels (normal/cataract) to class names
    """
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)
    
    # Process train split
    train_src = Path(source_dir) / 'train'
    if train_src.exists():
        print(f"\nImporting train split from {train_src}...")
        for label, class_name in label_mapping.items():
            label_dir = train_src / label
            if label_dir.exists():
                class_output_dir = os.path.join(output_train_dir, class_name)
                os.makedirs(class_output_dir, exist_ok=True)
                
                for img_file in label_dir.glob('*'):
                    if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        dst_path = os.path.join(class_output_dir, img_file.name)
                        shutil.copy2(img_file, dst_path)
                
                count = len(os.listdir(class_output_dir))
                print(f"  {class_name}: {count} images")
    
    # Process test split
    test_src = Path(source_dir) / 'test'
    if test_src.exists():
        print(f"\nImporting test split from {test_src}...")
        for label, class_name in label_mapping.items():
            label_dir = test_src / label
            if label_dir.exists():
                class_output_dir = os.path.join(output_test_dir, class_name)
                os.makedirs(class_output_dir, exist_ok=True)
                
                for img_file in label_dir.glob('*'):
                    if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        dst_path = os.path.join(class_output_dir, img_file.name)
                        shutil.copy2(img_file, dst_path)
                
                count = len(os.listdir(class_output_dir))
                print(f"  {class_name}: {count} images")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Import cataract datasets")
    parser.add_argument("--nuclear", help="Path to Nuclear dataset root (containing 01, 02, ... folders)")
    parser.add_argument("--processed", help="Path to processed dataset (containing train/, test/ folders)")
    parser.add_argument("--original-out", default="./datasets/original_images", help="Output for raw images")
    parser.add_argument("--train-out", default="./datasets/train", help="Output for training split")
    parser.add_argument("--test-out", default="./datasets/test", help="Output for test split")
    parser.add_argument("--flatten-nuclear", action="store_true", default=True, help="Flatten Nuclear dataset hierarchy")
    
    args = parser.parse_args()
    
    if args.nuclear:
        import_nuclear_dataset(
            args.nuclear,
            args.original_out,
            flatten=args.flatten_nuclear
        )
    
    if args.processed:
        import_processed_dataset(
            args.processed,
            args.train_out,
            args.test_out
        )
    
    print("\n=== Import Complete ===")
    print(f"Original images: {args.original_out}")
    print(f"Train split: {args.train_out}")
    print(f"Test split: {args.test_out}")
