import numpy as np
try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False
from ..base import Tensor
from ..base import as_Tensor
from ..base.functions import get_array_module
# from .meters import TimeMeter, AverageMeter
from ..base import Config

import sys
import time

def progress_bar(current, total, epoch, loss, acc, width=30): # 进度条
    """current: 当前已处理样本数；total: 总样本数"""
    percent = current / total
    filled = int(width * percent)
    bar = '#' * filled + '.' * (width - filled)
    sys.stdout.write(
        f'\rEpoch {epoch+1:3d} |{bar}| {percent*100:5.1f}% '
        f' | loss={loss:.4f} | acc={acc:.3f}'
    )
    sys.stdout.flush()


def _is_multilabel(y_hat, yb):
    return yb.ndim > 1 and y_hat.ndim == yb.ndim and y_hat.shape == yb.shape


def _is_regression(y_hat, yb, loss_fn=None):
    if loss_fn is not None:
        loss_name = getattr(loss_fn, "__name__", "")
        if loss_name == "meanSquaredError":
            return True
    return y_hat.ndim > 1 and y_hat.shape[-1] == 1 and yb.ndim == 1


def _batch_accuracy(y_hat, yb, y_true_cls, loss_fn=None):
    if _is_regression(y_hat, yb, loss_fn=loss_fn):
        return None, None

    if _is_multilabel(y_hat, yb):
        y_pred = (y_hat.data > 0.5).astype(np.int32)
        y_true = yb.data.astype(np.int32)
        batch_acc = np.all(y_pred == y_true, axis=1).mean()
        return y_pred, y_true, batch_acc

    if y_hat.ndim > 1:
        y_pred = y_hat.argmax(axis=1)          # Tensor.argmax 已返回原始 ndarray
    else:
        y_pred = y_hat.data if hasattr(y_hat, 'data') else y_hat

    # 统一提取底层 ndarray，避免 cupy_array == Tensor 触发
    # CuPy 的 __array_ufunc__ 协议（GPU 上不回退到反射运算符）
    y_true = y_true_cls.data if hasattr(y_true_cls, 'data') else y_true_cls

    batch_acc = (y_pred == y_true).mean()
    return y_pred, y_true, batch_acc

class Trainer:
    def __init__(self, model, loss_fn, optimizer, visualizer=None, enable_early_stop=False,
                 on_epoch_end=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self._epoch = 0
        self.visualizer = visualizer
        self._on_epoch_end = on_epoch_end
        self._stop_requested = False

        # 早停
        self.enable_early_stop = enable_early_stop
        self._early_stop_initialized = False

    def request_stop(self):
        """从外部请求停止训练（用于 Web API 中途中断）。"""
        self._stop_requested = True

    def init_early_stop(self, patience=5, mode='loss', min_delta=0.0, restore_best_weights=True):
        """
        初始化早停机制
        
        参数:
            patience: 容忍指标不提升的最大epoch数
            mode: 监控指标类型，'loss'表示越小越好，'acc'表示越大越好
            min_delta: 认为有改善的最小阈值，小于此值的变化不算改善
            restore_best_weights: 早停时是否恢复到最佳模型权重
        """
        if mode not in ['loss', 'acc']:
            raise ValueError(f"mode must be 'loss' or 'acc', got {mode}")
        
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_val_loss = np.inf
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.best_weights = None
        self._early_stop_initialized = True

    def _save_best_weights(self):
        """保存当前模型权重到最佳权重缓存"""
        self.best_weights = {}
        for param in self.model.get_params_list():
            if param.data is not None:
                self.best_weights[id(param)] = param.data.copy()
    
    def _restore_best_weights(self):
        """从最佳权重缓存恢复模型权重"""
        if self.best_weights is None:
            return
        
        for param in self.model.get_params_list():
            if param.data is not None and id(param) in self.best_weights:
                param.data[:] = self.best_weights[id(param)]

    def _check_early_stop(self, loss, acc, epoch, verbose):
        """
        检查是否应该早停
        
        返回:
            bool: True表示应该停止训练
        """
        if not self.enable_early_stop:
            return False
        
        if not self._early_stop_initialized:
            if verbose:
                print("Warning: Early stopping enabled but not initialized. Call init_early_stop() first.")
            return False
        
        improved = False
        
        if self.mode == 'loss':
            if loss < self.best_val_loss - self.min_delta:
                improved = True
                self.best_val_loss = loss
                if verbose:
                    print(f"  -> Validation loss improved to {loss:.4f}")
        elif self.mode == 'acc':
            if acc > self.best_val_acc + self.min_delta:
                improved = True
                self.best_val_acc = acc
                if verbose:
                    print(f"  -> Validation accuracy improved to {acc:.4f}")
        
        if improved:
            self.epochs_no_improve = 0
            if self.restore_best_weights:
                self._save_best_weights()
        else:
            self.epochs_no_improve += 1
            if verbose:
                print(f"  -> No improvement for {self.epochs_no_improve} epoch(s)")
        
        if self.epochs_no_improve >= self.patience:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best {'loss' if self.mode == 'loss' else 'accuracy'}: "
                      f"{self.best_val_loss:.4f if self.mode == 'loss' else self.best_val_acc:.4f}")
                print(f"{'='*60}")
            
            if self.restore_best_weights and self.best_weights is not None:
                if verbose:
                    print("Restoring best model weights...")
                self._restore_best_weights()
            
            return True
        
        return False


    def fit(self, train_loader, val_loader, epochs=10, batch_size=32, 
                verbose=True, device='cpu', start_epoch=0, checkpoint_path=None, serializer=None):
        """
            only accept DataLoader object
            if you need to split train/val dataset, please do it outside this function
        """
        self.model = self.model.to(device)
        
        if verbose:
            # 从 model 中随机抓取一个参数判断其类型以确认训练设备
            for param in self.model.params():
                if param.data is not None:
                    if has_cupy and isinstance(param.data, cp.ndarray):
                        print(f"using cuda to train")
                    else:
                        print(f"using cpu to train")
                    break

        for epoch in range(start_epoch, epochs):
            if verbose:
                print(f"======================= Epoch #{epoch+1}/{epochs} - Start training =======================")
            self._epoch = epoch
            tic = time.time()

            self._one_step(train_loader, batch_size=batch_size, training=True, verbose=verbose, device=device)

            # self.loss_meter.reset()
            # self.acc_meter.reset()

            loss, acc = self._one_step(val_loader, batch_size=len(val_loader.dataset), training=False, verbose=False, device=device)
            if verbose:
                print(f">>>>>>>>>>> Epoch loss: {loss:.4f} - Epoch acc: {acc:.4f}")
            toc = time.time()
            epoch_time = toc - tic
            if verbose:
                print(f"Time cost: {epoch_time:.2f} seconds")
            
            # 使用visualizer更新epoch时间
            if self.visualizer is not None:
                self.visualizer.update_epoch(epoch_time)
            
            # 保存checkpoint
            if checkpoint_path and serializer:
                serializer.save_checkpoint(self.model, self.optimizer, epoch, checkpoint_path)
                if verbose:
                    print(f"Checkpoint saved to {checkpoint_path}")
            
            # epoch 结束回调（Web 训练进度推送）
            if self._on_epoch_end is not None:
                train_loss_ep, train_acc_ep = self._one_step(
                    train_loader, batch_size=batch_size, training=False, verbose=False, device=device
                ) if False else (0.0, 0.0)  # 不重复跑训练集，用 val 指标近似
                self._on_epoch_end({
                    "epoch": epoch + 1,
                    "train_loss": float(loss),
                    "val_loss": float(loss),
                    "train_acc": float(acc),
                    "val_acc": float(acc),
                })

            # 早停
            if self._check_early_stop(loss, acc, epoch, verbose):
                break

            # 外部停止信号
            if self._stop_requested:
                if verbose:
                    print("\nTraining stopped by external request.")
                break

    def _one_step(self, data_loader, batch_size=32, training=True, verbose=True, device='cpu'):
        loss_sum, acc_sum, sample_num = 0., 0, 0
        y_true_list = []
        y_pred_list = []
        
        for batch_idx, (Xb, yb) in enumerate(data_loader):
            Xb = as_Tensor(Xb)
            yb = as_Tensor(yb)
            Xb = Xb.to(device)
            yb = yb.to(device)
            y_hat = self.model(Xb)

            # 兼容单标签分类与多标签分类
            if yb.ndim > 1:
                y_true_cls = yb.argmax(axis=1)
                if _is_multilabel(y_hat, yb):
                    # 多标签任务（如 SigmoidWithLoss）
                    y_target = yb
                else:
                    # 单标签分类 one-hot -> class index
                    y_target = y_true_cls
            else:
                y_true_cls = yb
                y_target = yb

            loss = self.loss_fn(y_hat, y_target)

            batch_acc_info = _batch_accuracy(y_hat, yb, y_true_cls, loss_fn=self.loss_fn)
            if batch_acc_info[0] is None:
                y_pred = y_hat.data
                y_true = y_true_cls.data if hasattr(y_true_cls, "data") else y_true_cls
                batch_acc = np.nan
            else:
                y_pred, y_true, batch_acc = batch_acc_info

            if training:
                self.model.cleargrads()
                loss.backward()
                '''
                    changed update to step
                '''
                self.optimizer.step()
            
            # 收集预测结果和真实标签，用于绘制混淆矩阵
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            
            loss_sum += loss.data * len(Xb)
            if not np.isnan(batch_acc):
                acc_sum += batch_acc * len(Xb)
            sample_num += len(Xb)

            # 使用visualizer更新指标
            if self.visualizer is not None:
                if not np.isnan(batch_acc):
                    if training:
                        self.visualizer.update_train(loss.data, batch_acc, batch_size=len(Xb))
                    else:
                        self.visualizer.update_val(loss.data, batch_acc, batch_size=len(Xb))
            
            # self.loss_meter.update(loss.data)
            # self.acc_meter.update(acc_sum / sample_num)
            
            if verbose:
                display_acc = (acc_sum / sample_num) if sample_num > 0 and not np.isnan(acc_sum) else np.nan
                progress_bar(batch_idx * batch_size + len(Xb), len(data_loader.dataset), self._epoch, loss.data, display_acc)
        if verbose:
            sys.stdout.write('\n')
        
        # 计算epoch级别的指标
        epoch_loss = loss_sum / sample_num
        epoch_acc = acc_sum / sample_num if acc_sum != 0 else (np.nan if _is_regression(y_hat, yb, loss_fn=self.loss_fn) else 0.0)
        
        # 更新visualizer的预测结果
        if self.visualizer is not None and not training:
            # 只在验证时更新预测结果，因为训练集可能很大
            for yt, yp in zip(y_true_list, y_pred_list):
                self.visualizer.update_predictions(yt, yp)
        
        return epoch_loss, epoch_acc 

class Evaluator:
    def __init__(self, model, loss_fn, visualizer=None):
        self.model = model
        self.loss_fn = loss_fn
        self.visualizer = visualizer
        # self.loss_meter = AverageMeter('Loss')
        # self.acc_meter = AverageMeter('Acc')
        # self.time_meter = TimeMeter()

    def evaluate(self, data_loader, batch_size=32, verbose=True, device='cpu'):
        loss_sum, acc_sum, sample_num = 0., 0, 0
        y_true_list = []
        y_pred_list = []
        
        if verbose:
            print(f"======================= Start evaluation =======================")

        self.model = self.model.to(device)
        for batch_idx, (Xb, yb) in enumerate(data_loader):
            Xb = as_Tensor(Xb)
            yb = as_Tensor(yb)
            Xb = Xb.to(device)
            yb = yb.to(device)

            with Config.using_config('train', False):
                y_hat = self.model(Xb)

                if yb.ndim > 1:
                    y_true_cls = yb.argmax(axis=1)
                    if _is_multilabel(y_hat, yb):
                        y_target = yb
                    else:
                        y_target = y_true_cls
                else:
                    y_true_cls = yb
                    y_target = yb

                loss = self.loss_fn(y_hat, y_target)

            batch_acc_info = _batch_accuracy(y_hat, yb, y_true_cls, loss_fn=self.loss_fn)
            if batch_acc_info[0] is None:
                y_pred = y_hat.data
                y_true = y_true_cls.data if hasattr(y_true_cls, "data") else y_true_cls
                batch_acc = np.nan
            else:
                y_pred, y_true, batch_acc = batch_acc_info
            
            # 收集预测结果和真实标签
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            
            loss_sum += loss.data * len(Xb)
            if not np.isnan(batch_acc):
                acc_sum += batch_acc * len(Xb)
            sample_num += len(Xb)

            # 使用visualizer更新指标
            if self.visualizer is not None:
                if not np.isnan(batch_acc):
                    self.visualizer.update_val(loss.data, batch_acc, batch_size=len(Xb))
                self.visualizer.update_predictions(y_true, y_pred)
            
            # self.loss_meter.update(loss.data)
            # self.acc_meter.update(acc_sum / sample_num)
        
        # 计算评估结果
        loss = loss_sum / sample_num
        acc = acc_sum / sample_num if acc_sum != 0 else (np.nan if _is_regression(y_hat, yb, loss_fn=self.loss_fn) else 0.0)
        
        return loss, acc

