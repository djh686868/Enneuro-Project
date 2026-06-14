"""完整模拟训练线程，捕获 cp 错误的完整 traceback"""
import sys, traceback
sys.path.insert(0, 'D:/Undergraduate/WZH/EnNeuro/tmp/Enneuro-Project/code')

from web_server.services.model_registry import build_model_from_config
from web_server.services.dataset_manager import (
    register_dataset, load_dataset_for_training, _detect_and_load_dataset
)
from eneuro.train.trainer import Trainer
from eneuro.nn.loss import SoftmaxWithLoss
from eneuro.nn.optim import Adam

MNIST_PATH = 'D:/Undergraduate/WZH/EnNeuro/tmp/Enneuro-Project/code/tests/testdata/MNIST_data'

# 注册数据集
ds_id = register_dataset('mnist', MNIST_PATH)
print(f'dataset_id: {ds_id}')

# 构建模型
_, model = build_model_from_config({'preset': 'LeNet'})
print(f'model: {type(model).__name__}')

# 加载数据
train_loader, val_loader = load_dataset_for_training(ds_id, batch_size=64, val_split=0.1)
print(f'train batches: {len(train_loader.dataset)}, val batches: {len(val_loader.dataset)}')

# 训练
optimizer = Adam(model.params(), lr=0.001)
loss_fn = SoftmaxWithLoss()

def on_epoch_end(m):
    print(f"  epoch {m['epoch']}: loss={m['val_loss']:.4f} acc={m['val_acc']:.4f}")

trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, on_epoch_end=on_epoch_end)

try:
    trainer.fit(train_loader, val_loader, epochs=1, batch_size=64, device='cpu', verbose=False)
    print('TRAINING OK')
except Exception as e:
    print('ERROR:', e)
    traceback.print_exc()
