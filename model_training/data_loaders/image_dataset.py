"""
Image datasets for deepfake detection training.

- InMemoryImageDataset:   Pre-loads ALL tensors into a single contiguous tensor in RAM with automatic shard caching. (for faster training)
- InMemoryCLIPDataset:    Alias for InMemoryImageDataset for the Spatial Stream.
- InMemorySRMDataset:     Alias for InMemoryImageDataset for the Noise Stream.
- PrecomputedCLIPDataset: Loads individual .pt / image files for the Spatial Stream.
- PrecomputedSRMDataset:  Loads individual .pt / image files for the Noise Stream.
"""

import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _load_dir(dir_path, label):
    """
    Load all .pt / image files from a single class directory.
    """

    if not os.path.exists(dir_path):
        return [], []
    files = sorted(
        f for f in os.listdir(dir_path)
        if f.endswith(('.pt', '.jpg', '.jpeg', '.png'))
    )
    tensors, labels = [], []
    for f in tqdm(files, desc=os.path.basename(dir_path), leave=False):
        fp = os.path.join(dir_path, f)
        if f.endswith('.pt'):
            tensors.append(torch.load(fp, weights_only=True))
        else:
            from PIL import Image
            import torchvision.transforms.functional as TF
            tensors.append(TF.to_tensor(Image.open(fp).convert('RGB')))
        labels.append(label)
    return tensors, labels


class InMemoryImageDataset(Dataset):
    """
    Pre-loads every sample into a single contiguous tensor in RAM for faster training.
    """

    def __init__(self, root_dir, dtype=torch.float32):
        shard_path = os.path.join(root_dir, '_shard.pt')

        if os.path.exists(shard_path):
            print(f"Loading shard {shard_path} …")
            shard = torch.load(shard_path, weights_only=True)
            self.data = shard['data'].to(dtype)
            self.targets = shard['targets']
        else:
            print(f"Building shard from {root_dir} (one-time) …")
            t_fake, l_fake = _load_dir(os.path.join(root_dir, 'fake'), 0)
            t_real, l_real = _load_dir(os.path.join(root_dir, 'real'), 1)
            self.data = torch.stack(t_fake + t_real).to(dtype)
            self.targets = torch.tensor(l_fake + l_real, dtype=torch.long)
            print(f"Saving shard → {shard_path}")
            torch.save({'data': self.data, 'targets': self.targets}, shard_path)

        n_fake = (self.targets == 0).sum().item()
        n_real = (self.targets == 1).sum().item()
        mem_gb = self.data.element_size() * self.data.nelement() / 1e9
        print(f"  {len(self.targets)} samples | Fake: {n_fake}, Real: {n_real}")
        print(f"  shape={tuple(self.data.shape)}, dtype={self.data.dtype}, RAM={mem_gb:.1f} GB")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].item()


# Convenience aliases so both streams can use the same class
InMemoryCLIPDataset = InMemoryImageDataset
InMemorySRMDataset  = InMemoryImageDataset


# -------------------------------------------------------------------
# Legacy per-file datasets (kept for backward compatibility)
# -------------------------------------------------------------------

class PrecomputedCLIPDataset(Dataset):
    """
    Per-file CLIP dataset for the Spatial Stream.
    """

    def __init__(self, root_dir):
        self.samples = []
        self.targets = []

        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            fake_files = [
                os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                if f.endswith(('.pt', '.jpg', '.jpeg', '.png'))
            ]
            self.samples.extend(fake_files)
            self.targets.extend([0] * len(fake_files))

        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            real_files = [
                os.path.join(real_dir, f) for f in os.listdir(real_dir)
                if f.endswith(('.pt', '.jpg', '.jpeg', '.png'))
            ]
            self.samples.extend(real_files)
            self.targets.extend([1] * len(real_files))

        print(f"Loaded {len(self.samples)} precomputed CLIP tensors from {root_dir}")
        print(f"  Fake: {self.targets.count(0)}, Real: {self.targets.count(1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath = self.samples[idx]
        if filepath.endswith('.pt'):
            tensor = torch.load(filepath, weights_only=True)
        else:
            from PIL import Image
            import torchvision.transforms.functional as TF
            image = Image.open(filepath).convert('RGB')
            tensor = TF.to_tensor(image)
        return tensor, self.targets[idx]


class PrecomputedSRMDataset(Dataset):
    """
    Per-file SRM dataset for the Noise Stream.
    """

    def __init__(self, root_dir):
        self.samples = []
        self.targets = []

        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            fake_files = [
                os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                if f.endswith(('.pt', '.jpg', '.jpeg', '.png'))
            ]
            self.samples.extend(fake_files)
            self.targets.extend([0] * len(fake_files))

        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            real_files = [
                os.path.join(real_dir, f) for f in os.listdir(real_dir)
                if f.endswith(('.pt', '.jpg', '.jpeg', '.png'))
            ]
            self.samples.extend(real_files)
            self.targets.extend([1] * len(real_files))

        print(f"Loaded {len(self.samples)} precomputed RGB tensors from {root_dir}")
        print(f"  Fake (Class 0): {self.targets.count(0)}, Real (Class 1): {self.targets.count(1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath = self.samples[idx]
        if filepath.endswith('.pt'):
            tensor = torch.load(filepath, weights_only=True)
        else:
            from PIL import Image
            import torchvision.transforms.functional as TF
            image = Image.open(filepath).convert('RGB')
            tensor = TF.to_tensor(image)
        return tensor, self.targets[idx]
