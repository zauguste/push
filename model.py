import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path


def build_model(num_classes=4, backbone='mobilenet_v2', pretrained=True):
    if backbone == 'mobilenet_v2':
        base = models.mobilenet_v2(pretrained=pretrained)
        in_features = base.classifier[1].in_features
        # Replace classifier
        base.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return base
    else:
        raise ValueError('Unsupported backbone')


def train(model, train_loader: DataLoader, val_loader: DataLoader = None, epochs=10, lr=1e-4, device=None, save_path='model.pt'):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total, acc = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            acc += (preds == yb).sum().item()
        print(f"Epoch {epoch}/{epochs} - train acc: {acc/total:.4f}")
        if val_loader is not None:
            model.eval()
            vtotal, vacc = 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    preds = logits.argmax(dim=1)
                    vtotal += yb.size(0)
                    vacc += (preds == yb).sum().item()
            print(f"  val acc: {vacc/vtotal:.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


def convert_to_coreml(pytorch_model_path, input_shape=(1, 3, 224, 224), coreml_path='model.mlmodel'):
    try:
        import coremltools as ct
    except Exception:
        raise RuntimeError('coremltools is not installed in this environment')
    dummy_input = torch.randn(input_shape)
    # User must recreate model architecture and load state_dict before tracing
    # This function is a placeholder: adapt to your exact model definition.
    raise NotImplementedError('Implement model reconstruction and conversion with coremltools here')


if __name__ == '__main__':
    print('Model helpers: build_model, train, convert_to_coreml')
