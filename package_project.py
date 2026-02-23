"""
Package the cataract detection project into a zip file.
Includes: source code, models, datasets structure, docs, MLflow runs.
Excludes: venv, __pycache__, large binary environments.
"""

import zipfile
import os
from pathlib import Path
from datetime import datetime
import shutil


def should_skip(filepath):
    """Determine if a file/dir should be excluded from zip."""
    skip_patterns = [
        '.venv', '__pycache__', '.git', '.pytest_cache',
        '*.pyc', '*.pyo', '.DS_Store', 'Thumbs.db',
        '.egg-info', '.eggs', 'build', 'dist',
        '.vscode', '.idea', '*.egg'
    ]
    
    path = Path(filepath)
    
    # Skip hidden directories
    if path.name.startswith('.') and path.name not in ['.gitignore']:
        return True
    
    # Skip patterns
    for pattern in skip_patterns:
        if '*' in pattern:
            if path.match(pattern):
                return True
        else:
            if pattern in str(filepath):
                return True
    
    return False


def create_project_zip(output_file='cataract_detection.zip'):
    """Create a zip file of the entire project."""
    
    project_root = Path('.')
    
    print(f"📦 Packaging project into {output_file}...")
    print(f"Root: {project_root.absolute()}\n")
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        total_files = 0
        total_size = 0
        
        for item in project_root.rglob('*'):
            # Skip certain dirs/files
            if should_skip(item):
                continue
            
            # Skip very large files (>100MB)
            if item.is_file():
                try:
                    size = item.stat().st_size
                    if size > 100 * 1024 * 1024:  # 100MB
                        print(f"  ⊘ Skipped (too large): {item.relative_to(project_root)}")
                        continue
                except:
                    continue
            
            # Add to zip
            try:
                arcname = item.relative_to(project_root)
                
                if item.is_file():
                    zipf.write(item, arcname)
                    total_files += 1
                    total_size += item.stat().st_size
                    print(f"  ✓ {arcname}")
                elif item.is_dir() and not any(item.iterdir()):
                    # Add empty directories
                    zipf.write(item, arcname + '/')
                    print(f"  ✓ {arcname}/ (empty)")
            except PermissionError:
                print(f"  ⚠ Skipped (permission): {arcname}")
            except Exception as e:
                print(f"  ⚠ Skipped (error): {item.relative_to(project_root)} - {e}")
    
    # Print summary
    zip_size = os.path.getsize(output_file)
    print(f"\n" + "="*70)
    print(f"✅ Package created: {output_file}")
    print(f"   Files: {total_files}")
    print(f"   Uncompressed: {total_size / (1024**2):.1f} MB")
    print(f"   Compressed: {zip_size / (1024**2):.1f} MB")
    print(f"   Ratio: {100 * zip_size / total_size:.1f}%")
    print("="*70)
    
    print(f"\n📋 Contents Summary:\n")
    print("  ✓ Source code (train.py, predict.py, preprocess.py, etc.)")
    print("  ✓ Trained models (checkpoints/)")
    print("  ✓ Training history and metrics (checkpoints/history.json)")
    print("  ✓ MLflow runs (mlruns/)")
    print("  ✓ Dataset structure (datasets/ - images not included if >100MB)")
    print("  ✓ Configuration and README")
    print("  ✓ All Python modules and utilities")
    
    print(f"\n🚀 To extract and use:")
    print(f"   unzip {output_file}")
    print(f"   cd [extracted_folder]")
    print(f"   python -m venv .venv")
    print(f"   .\.venv\Scripts\pip install -r requirements.txt")
    print(f"   python train.py --help")
    
    return output_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Package project into zip")
    parser.add_argument("--output", default="cataract_detection.zip", help="Output zip filename")
    parser.add_argument("--timestamp", action="store_true", help="Add timestamp to filename")
    
    args = parser.parse_args()
    
    if args.timestamp:
        name, ext = os.path.splitext(args.output)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{name}_{timestamp}{ext}"
    
    create_project_zip(args.output)
