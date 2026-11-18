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


def register_dataset(name: str, path: str, description: str = "",
                     img_size: tuple = None) -> str:
    dataset_id = str(uuid.uuid4())[:8]
    _datasets[dataset_id] = {
        "dataset_id": dataset_id,
        "name": name,
        "path": path,
        "description": description,
        "img_size": img_size,   # None = 原图；(W, H) = resize
    }
    return dataset_id


def list_datasets() -> list[dict]:
    return list(_datasets.values())


class FolderImageDataset(Dataset):
    """
    从目录结构加载图像分类数据集。
    目录格式：root/class_a/img1.jpg, root/class_b/img2.png ...
    也支持 root/ 下直接存放图片（单类）。
    channels=None 时自动检测（彩色→3，灰度→1）；否则强制指定通道数。
    """
    def __init__(self, root: str, img_size=None, channels=None, transform=None):
        self.root = root
        self.img_size = img_size   # None = 保持原图尺寸，否则 (W, H) resize
        self.channels = channels   # None=auto, 1=gray, 3=rgb
        self._samples: list[tuple[str, int]] = []
        self._classes: list[str] = []
        super().__init__(transform=transform)

    def prepare(self):
        import cv2
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

        # 自动检测通道数（若未指定）：读取第一张图判断是否为彩色
        if self.channels is None and self._samples:
            probe = cv2.imread(self._samples[0][0], cv2.IMREAD_COLOR)
            if probe is not None:
                b, g, r = probe[:, :, 0], probe[:, :, 1], probe[:, :, 2]
                self.channels = 1 if (b == g).all() and (g == r).all() else 3
            else:
                self.channels = 1

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        import cv2
        path, label = self._samples[index]

        ch = self.channels if self.channels is not None else 1
        if ch == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                h = self.img_size[1] if self.img_size else 32
                w = self.img_size[0] if self.img_size else 32
                img = np.zeros((h, w), dtype=np.uint8)
            if self.img_size is not None:
                img = cv2.resize(img, self.img_size)
            x = img.astype(np.float32) / 255.0
            x = x[np.newaxis, :, :]   # (1, H, W)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                h = self.img_size[1] if self.img_size else 32
                w = self.img_size[0] if self.img_size else 32
                img = np.zeros((h, w, 3), dtype=np.uint8)
            if self.img_size is not None:
                img = cv2.resize(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = img.astype(np.float32) / 255.0
            x = x.transpose(2, 0, 1)   # (3, H, W)

        return x, np.array(label, dtype=np.int32)


def _detect_and_load_dataset(path: str, img_size=None):
    """
    自动检测数据集格式并返回 Dataset 实例。
    img_size: None = 原图；(W, H) = resize 到指定尺寸。
    """
    # ── 格式 1：目录下含 .pkl 文件 ──
    pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
    if pkl_files:
        return MNISTPickleDataset(pkl_path=os.path.join(path, pkl_files[0]),
                                  out_size=img_size)

    # ── 格式 2 & 3：图片文件夹 ──
    return FolderImageDataset(root=path, img_size=img_size)


class MNISTPickleDataset(Dataset):
    """
    加载 MNIST 风格的 pkl 文件。
    支持以下 key 格式（自动探测）：
      - {'train_img': (N,784), 'train_label': (N,), 'test_img': ..., 'test_label': ...}
      - {'x_train': (N,784), 'y_train': (N,), ...}
      - ((x_train, y_train), (x_test, y_test))  ← Keras 格式
    合并 train + test，由上层 val_split 负责划分。
    图像统一 resize 到 out_size（默认 32×32），与 FolderImageDataset 一致，
    避免 28×28 MNIST 送入为 32×32 设计的 LeNet 等模型时 FC 层维度不匹配。
    """
    def __init__(self, pkl_path: str, img_shape=(28, 28), out_size=None, transform=None):
        self.pkl_path = pkl_path
        self.img_shape = img_shape
        self.out_size  = out_size   # None = 不 resize，保持 img_shape 原始大小
        self._images: np.ndarray = None   # (N, 1, H, W) float32
        self._labels: np.ndarray = None   # (N,) int32
        self._classes: list[str] = [str(i) for i in range(10)]
        super().__init__(transform=transform)

    def prepare(self):
        import pickle
        with open(self.pkl_path, 'rb') as f:
            raw = pickle.load(f, encoding='bytes' if self._is_bytes_pkl() else 'ASCII')

        imgs_list, labels_list = [], []

        if isinstance(raw, dict):
            # 尝试常见 key 组合
            key_map = {
                'train_img':   'train_label',
                'x_train':     'y_train',
                b'train_img':  b'train_label',
                b'x_train':    b'y_train',
            }
            test_key_map = {
                'train_img':   ('test_img',   'test_label'),
                'x_train':     ('x_test',     'y_test'),
                b'train_img':  (b'test_img',  b'test_label'),
                b'x_train':    (b'x_test',    b'y_test'),
            }
            for img_key, lbl_key in key_map.items():
                if img_key in raw:
                    imgs_list.append(raw[img_key])
                    labels_list.append(raw[lbl_key])
                    # 附加测试集（如果存在）
                    ti_key, tl_key = test_key_map[img_key]
                    if ti_key in raw:
                        imgs_list.append(raw[ti_key])
                        labels_list.append(raw[tl_key])
                    break
            if not imgs_list:
                raise ValueError(f"Unrecognized pkl dict keys: {list(raw.keys())[:5]}")

        elif isinstance(raw, (list, tuple)) and len(raw) == 2:
            # Keras 格式：((x_train, y_train), (x_test, y_test))
            (x_tr, y_tr), (x_te, y_te) = raw
            imgs_list   = [x_tr, x_te]
            labels_list = [y_tr, y_te]
        else:
            raise ValueError(f"Unrecognized pkl format: {type(raw)}")

        imgs   = np.concatenate(imgs_list,   axis=0).astype(np.float32) / 255.0
        labels = np.concatenate(labels_list, axis=0).astype(np.int32)

        # reshape: (N, 784) → (N, 1, 28, 28) 或 (N, 28, 28) → (N, 1, 28, 28)
        if imgs.ndim == 2:
            H, W = self.img_shape
            imgs = imgs.reshape(-1, 1, H, W)
        elif imgs.ndim == 3:
            imgs = imgs[:, np.newaxis, :, :]

        # 可选 resize
        if self.out_size is not None:
            oH, oW = self.out_size
            if imgs.shape[2] != oH or imgs.shape[3] != oW:
                import cv2
                resized = np.empty((len(imgs), 1, oH, oW), dtype=np.float32)
                for i, img in enumerate(imgs):
                    resized[i, 0] = cv2.resize(img[0], (oW, oH), interpolation=cv2.INTER_LINEAR)
                imgs = resized

        self._images = imgs
        self._labels = labels

    def _is_bytes_pkl(self) -> bool:
        try:
            import pickle
            with open(self.pkl_path, 'rb') as f:
                pickle.load(f)
            return False
        except Exception:
            return True

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        return self._images[index], self._labels[index]

    @property
    def classes(self):
        return self._classes


def load_dataset_for_training(dataset_id: str, batch_size: int, val_split: float):
    meta = _datasets.get(dataset_id)
    if meta is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    dataset = _detect_and_load_dataset(meta["path"], img_size=meta.get("img_size"))
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
