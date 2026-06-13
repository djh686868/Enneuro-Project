import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'code'))

import os
import numpy as np
import matplotlib.pyplot as plt

from eneuro.nn.optim import Adam
from eneuro.nn.loss import meanSquaredError
from eneuro.train import Trainer
from eneuro.utils import Visualizer
from eneuro.data.dataloader import DataLoader

from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


batch_size = 32
total_epochs = 1
lr =1e-4

model = ResNet18AutoDrive()

from eneuro.utils.serializer import Serializer
serializer = Serializer()
script_dir = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(script_dir, "results")
os.makedirs(save_folder, exist_ok=True)
model_path = os.path.join(save_folder, "model_gpu.json")

if os.path.exists(model_path):
    print(f"Loading pre-trained model from {model_path}...")
    serializer.load(model, model_path)
    print("Model loaded successfully. Starting fine-tuning...")
else:
    print("No pre-trained model found, starting training from scratch")

optimizer = Adam(model.params(), lr=lr)

loss_fn = meanSquaredError

visualizer = Visualizer(num_classes=1)

train_dataset = AutoDriveDataset(mode="train", transform=preprocess_image)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

val_dataset = AutoDriveDataset(mode="val", transform=preprocess_image)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

trainer = Trainer(model, loss_fn, optimizer, visualizer)

trainer.fit(
    train_loader,
    val_loader,
    epochs=total_epochs,
    batch_size=batch_size,
    verbose=True,
    device='cuda'
)

print(f"Saving model to {save_folder}/model_gpu.json...")
serializer.save(model, save_folder + "/model_gpu.json")
print("Model saved successfully.")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Training Visualization (Regression)', fontsize=16)

axes[0].plot(visualizer.train_loss, label='Train Loss')
axes[0].plot(visualizer.val_loss, label='Val Loss')
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(visualizer.epoch_times)
axes[1].set_title('Time Consumption per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Time (seconds)')
axes[1].grid(True)

axes[2].plot(visualizer.train_loss, label='Train Loss')
axes[2].plot(visualizer.val_loss, label='Val Loss')
axes[2].set_title('Loss Curve (Zoomed)')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss')
axes[2].set_ylim([0, 0.05])
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig(os.path.join(save_folder, "training_curves.png"), dpi=150, bbox_inches='tight')
print(f"Training curves saved to {save_folder}/training_curves.png")
plt.show()
plt.close()
