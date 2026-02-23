import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class CataractDataset(Dataset):
    """Simple image dataset for cataract grading.

    Expects a list of (image_path, label) tuples or a directory structure where
    `labels.csv` or a mapping is provided externally.
    """
    def __init__(self, samples, transform=None):
        # samples: list of (path, int_label)
        self.samples = list(samples)
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, int(label)

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CataractDataset(Dataset):
    def __init__(self, root_dir, transform=None, extensions=(".jpg", ".jpeg", ".png")):
        self.root_dir = root_dir
        self.samples = []
        for label_name in sorted(os.listdir(root_dir)):
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            for fn in sorted(os.listdir(label_dir)):
                if any(fn.lower().endswith(ext) for ext in extensions):
                    self.samples.append((os.path.join(label_dir, fn), label_name))
        self.label_map = {name: i for i, name in enumerate(sorted({l for _, l in self.samples}))}
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_name = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = self.label_map[label_name]
        return img, label
