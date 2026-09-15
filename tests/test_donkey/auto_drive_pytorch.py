# -*- coding: utf-8 -*-
import os
import numpy as np
import gymnasium as gym
import gym_donkeycar
import torch
import torch.nn as nn
from pathlib import Path


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet18AutoDrive(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18AutoDrive, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(blocks - 1):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    env = gym.make("donkey-generated-roads-v0")
    obv = env.reset()

    model = ResNet18AutoDrive(num_classes=1).to(device)
    script_dir = Path(__file__).resolve().parent
    model_path = str(script_dir / "results" / "model_pytorch.pt")
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Model not found at {model_path}")
        return

    action = np.array([0, 0.2])
    frame, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print("Starting autonomous driving simulation...")
    for t in range(2500):
        img = frame.astype(np.float32) / 255.0
        img = img.transpose([2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img_tensor = torch.from_numpy(img).float().to(device)

        with torch.no_grad():
            prelabel = model(img_tensor)
            steering_angle = prelabel[0, 0].item()

        factor = 1.5
        action = np.array([steering_angle * factor, 0.2])
        frame, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if t % 100 == 0:
            print(f"Step {t}: steering_angle={steering_angle:.4f}, reward={reward:.4f}")

        if done:
            print(f"Episode finished at step {t}")
            break

    obv = env.reset()
    env.close()
    print("Simulation completed.")


if __name__ == '__main__':
    main()