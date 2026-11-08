"""MNIST data and LeNet architecture adapted from code/tests/test_lenet.py.

The test script's LeNet replaces Layer._params with Parameter objects and has
lazy weights; this module keeps the same layer dimensions while making every
weight available to the parameter server at construction time.
"""

import pickle
from pathlib import Path

import numpy as np

from ..base import functions as F
from ..nn.module import Module, Conv2d, Linear
from ..nn.optim import SGD


DEFAULT_MNIST_PATH = Path(__file__).resolve().parents[3] / 'tests' / 'testdata' / 'MNIST_data' / 'mnist.pkl'


class MNISTTrainDataset:
    def __init__(self, path=DEFAULT_MNIST_PATH, limit=None):
        with Path(path).open('rb') as stream:
            data = pickle.load(stream)
        images, labels = data['train_img'], data['train_label']
        if limit is not None:
            if limit <= 0 or limit > len(labels):
                raise ValueError('limit must be within the MNIST training set')
            images, labels = images[:limit], labels[:limit]
        images = np.asarray(images, dtype=np.float32)
        if images.max() > 1:
            images /= 255.0
        self.images = images.reshape(-1, 1, 28, 28)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class LeNetMNIST(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(6, kernel_size=5, in_channels=1)
        self.conv2 = Conv2d(16, kernel_size=5, in_channels=6)
        self.fc1 = Linear(120, in_size=16 * 4 * 4)
        self.fc2 = Linear(84, in_size=120)
        self.fc3 = Linear(10, in_size=84)

    def forward(self, x):
        x = F.pooling(F.relu(self.conv1(x)), 2, 2)
        x = F.pooling(F.relu(self.conv2(x)), 2, 2)
        x = F.flatten(x)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return self.fc3(x)


def make_lenet():
    np.random.seed(42)
    return LeNetMNIST()


def make_mnist_dataset(limit=None, path=DEFAULT_MNIST_PATH):
    return MNISTTrainDataset(path, limit)


def make_optimizer(parameters):
    return SGD(parameters, lr=0.01)
