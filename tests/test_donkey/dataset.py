import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'code'))

import numpy as np
import cv2

from eneuro.data.dataset import Dataset


def preprocess_image(img):
    """图像预处理：归一化并调整通道"""
    img = img.astype(np.float32) / 255.0
    img = img.transpose([2, 0, 1])
    return img


class AutoDriveDataset(Dataset):
    """数据集加载器"""

    def __init__(self, mode, transform=None):
        """
        :参数 mode: 'train' 或者 'val'
        :参数 transform: 图像预处理方式
        """
        self.mode = mode.lower()
        self.transform = transform
        assert self.mode in {"train", "val"}
        
        current_dir = Path(__file__).resolve().parent
        if self.mode == "train":
            file_path = str(current_dir / "train.txt")
        else:
            file_path = str(current_dir / "val.txt")
        
        self.file_list = list()
        with open(file_path, "r") as f:
            files = f.readlines()
            for file in files:
                if file.strip() is None:
                    continue
                self.file_list.append([file.split(" ")[0], float(file.split(" ")[1])])
        
        self.data = list(range(len(self.file_list)))
        self.label = [self.file_list[i][1] for i in range(len(self.file_list))]
        self.prepare()

    def __getitem__(self, index):
        img = cv2.imread(self.file_list[index][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        label = np.array([self.file_list[index][1]], dtype=np.float32)
        return img, label
