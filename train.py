"""
Training script for cataract detection (Healthy vs. Severe).
Loads pre-split datasets from datasets/train and datasets/test directories.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import json
from datetime import datetime
import mlflow
import mlflow.pytorch


class CataractClassificationDataset(Dataset):
    """Load images from class-organized directory structure."""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory with subdirectories for each class (e.g., Healthy/, Severe/)
            transform: Torchvision transforms to apply
        """
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Discover classes from subdirectories
        class_dirs = sorted([d for d in Path(root_dir).iterdir() if d.is_dir()])
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            # Find all images in this class
            for img_file in class_dir.glob('*'):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    self.samples.append((str(img_file), idx))
        
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.samples)} images from {len(class_dirs)} classes")
        for class_name, idx in self.class_to_idx.items():
            count = sum(1 for _, label in self.samples if label == idx)
            print(f"  {class_name}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank tensor on error
            img = torch.zeros(3, 224, 224)
        
        return img, label


def build_model(num_classes=2, pretrained=True, dropout=0.5):
    """Build MobileNetV2 for cataract classification."""
    model = models.mobilenet_v2(pretrained=pretrained)
    in_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes)
    )
    return model


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx+1}: loss={loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def train(
    train_dir='datasets/train',
    test_dir='datasets/test',
    num_epochs=20,
    batch_size=32,
    learning_rate=1e-4,
    device=None,
    save_dir='./checkpoints',
    experiment_name='cataract_detection',
    run_name=None
):
    """
    Full training pipeline.
    
    Args:
        train_dir: Path to training data (with class subdirectories)
        test_dir: Path to test data (with class subdirectories)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        device: torch device (auto-detect if None)
        save_dir: Where to save checkpoints
    """
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup MLflow
    mlflow.set_experiment(experiment_name)
    run_name = run_name or f"cataract_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=run_name)
    
    print(f"MLflow experiment: {experiment_name}")
    print(f"MLflow run: {run_name}\n")
    
    # Load datasets
    print("Loading training dataset...")
    train_dataset = CataractClassificationDataset(train_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print("\nLoading test dataset...")
    test_dataset = CataractClassificationDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    num_classes = len(train_dataset.class_to_idx)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {list(train_dataset.class_to_idx.keys())}\n")
    
    # Build model
    print("Building model...")
    model = build_model(num_classes=num_classes, pretrained=True)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Log parameters to MLflow
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("model_architecture", "MobileNetV2")
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss_function", "CrossEntropyLoss")
    mlflow.log_param("device", str(device))
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...\n")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_test_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"  Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Log metrics to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric("test_acc", test_acc, step=epoch)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            checkpoint_path = os.path.join(save_dir, f'model_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'class_to_idx': train_dataset.class_to_idx,
                'idx_to_class': train_dataset.idx_to_class,
            }, checkpoint_path)
            print(f"  ✓ Best model saved to {checkpoint_path}")
        
        scheduler.step(test_loss)
        print()
    
    # Save final model
    final_path = os.path.join(save_dir, f'model_final.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': train_dataset.class_to_idx,
        'idx_to_class': train_dataset.idx_to_class,
    }, final_path)
    print(f"Final model saved to {final_path}")
    
    # Save training history
    history_path = os.path.join(save_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Log artifacts to MLflow
    mlflow.log_artifact(final_path, artifact_path="models")
    mlflow.log_artifact(os.path.join(save_dir, 'model_best.pt'), artifact_path="models")
    mlflow.log_artifact(history_path, artifact_path="metrics")
    
    # Log final model with MLflow PyTorch flavor
    mlflow.pytorch.log_model(model, "pytorch_model", registered_model_name=None)
    
    print("Artifacts logged to MLflow")
    
    # Summary
    print("\n" + "="*60)
    print(f"Best test accuracy: {best_test_acc:.4f} (Epoch {best_epoch})")
    print(f"Final test accuracy: {history['test_acc'][-1]:.4f}")
    print("="*60)
    
    # Log final metrics to MLflow
    mlflow.log_metric("best_test_accuracy", best_test_acc)
    mlflow.log_metric("final_test_accuracy", history['test_acc'][-1])
    mlflow.log_metric("best_epoch", best_epoch)
    
    mlflow.end_run()
    print(f"\nMLflow run completed: {run_name}")
    
    return model, history, train_dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train cataract classification model")
    parser.add_argument("--train-dir", default="datasets/train", help="Training data directory")
    parser.add_argument("--test-dir", default="datasets/test", help="Test data directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-dir", default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--experiment", default="cataract_detection", help="MLflow experiment name")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    
    args = parser.parse_args()
    
    model, history, dataset = train(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        experiment_name=args.experiment,
        run_name=args.run_name
    )
