"""数据集管理：注册、存储路径、构建 DataLoader。"""
import os
import uuid
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from eneuro.data.dataset import Dataset
from eneuro.data.dataloader import DataLoader
from eneuro.base import Tensor

_datasets: dict[str, dict] = {}
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "../../../uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def register_dataset(name: str, path: str, description: str = "") -> str:
    dataset_id = str(uuid.uuid4())[:8]
    _datasets[dataset_id] = {
        "dataset_id": dataset_id,
        "name": name,
        "path": path,
        "description": description,
    }
    return dataset_id


def list_datasets() -> list[dict]:
    return list(_datasets.values())


class FolderImageDataset(Dataset):
    """
    从目录结构加载图像分类数据集。
    目录格式：root/class_a/img1.jpg, root/class_b/img2.png ...
    也支持 root/ 下直接存放图片（单类）。
    """
    def __init__(self, root: str, img_size=(32, 32), transform=None):
        self.root = root
        self.img_size = img_size
        self._samples: list[tuple[str, int]] = []
        self._classes: list[str] = []
        super().__init__(transform=transform)

    def prepare(self):
        subdirs = [
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ]
        if subdirs:
            self._classes = sorted(subdirs)
            for label, cls in enumerate(self._classes):
                cls_dir = os.path.join(self.root, cls)
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        self._samples.append((os.path.join(cls_dir, fname), label))
        else:
            # 根目录下直接是图片，文件名格式 {idx}_{label}.jpg
            self._classes = ["unknown"]
            for fname in os.listdir(self.root):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    parts = fname.split("_")
                    try:
                        label = float(parts[-1].rsplit(".", 1)[0])
                        label_idx = 0
                    except ValueError:
                        label_idx = 0
                    self._samples.append((os.path.join(self.root, fname), label_idx))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        import cv2
        path, label = self._samples[index]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(self.img_size, dtype=np.uint8)
        img = cv2.resize(img, self.img_size)
        x = img.astype(np.float32) / 255.0
        x = x[np.newaxis, :, :]   # (1, H, W)
        return x, np.array(label, dtype=np.int32)


def load_dataset_for_training(dataset_id: str, batch_size: int, val_split: float):
    meta = _datasets.get(dataset_id)
    if meta is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    dataset = FolderImageDataset(root=meta["path"])
    n = len(dataset)
    if n == 0:
        raise ValueError("Dataset is empty")

    val_n = max(1, int(n * val_split))
    indices = np.random.permutation(n)
    val_idx = indices[:val_n].tolist()
    train_idx = indices[val_n:].tolist()

    class SubsetDataset(Dataset):
        def __init__(self, base, idx):
            self._base = base
            self._idx = idx
            super().__init__()
        def prepare(self): pass
        def __len__(self): return len(self._idx)
        def __getitem__(self, i):
            x, y = self._base[self._idx[i]]
            return Tensor(x), Tensor(y)

    train_ds = SubsetDataset(dataset, train_idx)
    val_ds   = SubsetDataset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    return train_loader, val_loader
