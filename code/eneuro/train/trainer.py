import numpy as np
from ..base import Tensor
from ..base import as_Tensor
# from .meters import TimeMeter, AverageMeter
from ..base import Config

import sys
import time

def progress_bar(current, total, epoch, loss, acc, width=30): # 进度条
    """current: 当前已处理样本数；total: 总样本数"""
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    sys.stdout.write(
        f'\rEpoch {epoch+1:3d} |{bar}| {percent*100:5.1f}% '
        f' | loss={loss:.4f} | acc={acc:.3f}'
    )
    sys.stdout.flush()

class Trainer:
    def __init__(self, model, loss_fn, optimizer, visualizer=None, enable_early_stop=False):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self._epoch = 0
        self.visualizer = visualizer
        # self.loss_meter = AverageMeter('Loss')
        # self.acc_meter = AverageMeter('Acc')
        # self.time_meter = TimeMeter()

        # 早停
        self.enable_early_stop = enable_early_stop

    def init_early_stop(self, patience=5, mode='loss'):
        self.patience = patience
        self.mode = mode
        self.best_val_loss = np.inf
        self.best_val_acc = 0
        self.epochs_no_improve = 0
        self.best_weights = None


    def fit(self, train_loader, val_loader, epochs=10, batch_size=32, 
                verbose=True, device='cpu'):
        """
            only accept DataLoader object
            if you need to split train/val dataset, please do it outside this function
        """
        # self.model.to(device)

        for epoch in range(epochs):
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
            
            # 早停
            # if self.enable_early_stop:
            #     if self.mode == 'loss':
            #         if loss < self.best_val_loss:
            #             self.best_val_loss = loss
            #             self.best_weights = self.model.get_params()
            #             self.epochs_no_improve = 0
            #         else:
            #             self.epochs_no_improve += 1
            #     elif self.mode == 'acc':
            #         if acc > self.best_val_acc:
            #             self.best_val_acc = acc
            #             self.best_weights = self.model.get_params()
            #             self.epochs_no_improve = 0
            #         else:
            #             self.epochs_no_improve += 1
            #     else:
            #         raise ValueError(f"Invalid mode: {self.mode}")
                
            #     if self.epochs_no_improve >= self.patience:
            #         print(f"Early stopping after {epoch+1} epochs")
            #         break

    def _one_step(self, data_loader, batch_size=32, training=True, verbose=True, device='cpu'):
        loss_sum, acc_sum, sample_num = 0., 0, 0
        y_true_list = []
        y_pred_list = []
        
        for batch_idx, (Xb, yb) in enumerate(data_loader):
            Xb = as_Tensor(Xb)
            yb = as_Tensor(yb)
            # Xb = Xb.to(device)
            # yb = yb.to(device)
            y_hat = self.model(Xb)

            # 兼容单标签分类与多标签分类
            if yb.ndim > 1:
                y_true_cls = yb.argmax(axis=1)
                if y_hat.ndim == yb.ndim and y_hat.shape == yb.shape:
                    # 多标签任务（如 SigmoidWithLoss）
                    y_target = yb
                else:
                    # 单标签分类 one-hot -> class index
                    y_target = y_true_cls
            else:
                y_true_cls = yb
                y_target = yb

            loss = self.loss_fn(y_hat, y_target)

            if y_hat.ndim > 1:
                if yb.ndim > 1 and y_hat.shape == yb.shape:
                    # 多标签预测：阈值化后逐元素比较
                    y_pred = (y_hat.data > 0.5).astype(np.int32)
                    y_true = yb.data.astype(np.int32)
                else:
                    y_pred = y_hat.argmax(axis=1)
                    y_true = y_true_cls
            else:
                y_pred = y_hat
                y_true = y_true_cls

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
            acc_sum += (y_pred == y_true).sum()
            sample_num += len(Xb)

            # 使用visualizer更新指标
            if self.visualizer is not None:
                batch_acc = (y_pred == y_true).sum() / len(Xb)
                if training:
                    self.visualizer.update_train(loss.data, batch_acc, batch_size=len(Xb))
                else:
                    self.visualizer.update_val(loss.data, batch_acc, batch_size=len(Xb))
            
            # self.loss_meter.update(loss.data)
            # self.acc_meter.update(acc_sum / sample_num)
            
            if verbose:
                progress_bar(batch_idx * batch_size + len(Xb), len(data_loader.dataset), self._epoch, loss.data, acc_sum / sample_num)
        if verbose:
            sys.stdout.write('\n')
        
        # 计算epoch级别的指标
        epoch_loss = loss_sum / sample_num
        epoch_acc = acc_sum / sample_num
        
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

        # self.model.to(device)
        for batch_idx, (Xb, yb) in enumerate(data_loader):
            Xb = as_Tensor(Xb)
            yb = as_Tensor(yb)
            # Xb = Xb.to(device)
            # yb = yb.to(device)

            with Config.using_config('train', False):
                y_hat = self.model(Xb)

                if yb.ndim > 1:
                    y_true_cls = yb.argmax(axis=1)
                    if y_hat.ndim == yb.ndim and y_hat.shape == yb.shape:
                        y_target = yb
                    else:
                        y_target = y_true_cls
                else:
                    y_true_cls = yb
                    y_target = yb

                loss = self.loss_fn(y_hat, y_target)

            if y_hat.ndim > 1:
                if yb.ndim > 1 and y_hat.shape == yb.shape:
                    y_pred = (y_hat.data > 0.5).astype(np.int32)
                    y_true = yb.data.astype(np.int32)
                else:
                    y_pred = y_hat.data.argmax(axis=1)
                    y_true = y_true_cls
            else:
                y_pred = y_hat.data
                y_true = y_true_cls
            
            # 收集预测结果和真实标签
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            
            loss_sum += loss.data * len(Xb)
            acc_sum += (y_pred == y_true).sum()
            sample_num += len(Xb)

            # 使用visualizer更新指标
            if self.visualizer is not None:
                batch_acc = (y_pred == y_true).sum() / len(Xb)
                self.visualizer.update_val(loss.data, batch_acc, batch_size=len(Xb))
                self.visualizer.update_predictions(y_true, y_pred)
            
            # self.loss_meter.update(loss.data)
            # self.acc_meter.update(acc_sum / sample_num)
        
        # 计算评估结果
        loss = loss_sum / sample_num
        acc = acc_sum / sample_num
        
        return loss, acc

        


'''
this is the original version of split_train_val

class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self._epoch = 0
        # self.loss_meter = AverageMeter('Loss')
        # self.acc_meter = AverageMeter('Acc')
        # self.time_meter = TimeMeter()

    def fit(self, data_loader, epochs=10, batch_size=32, 
                verbose=True, tr_val_split=0.8, device='cpu'):
        # self.model.to(device)
        X, y = data_loader.get_data()
        split_idx = int(len(X) * tr_val_split)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        for epoch in range(epochs):
            if verbose:
                print(f"======================= Epoch #{epoch+1}/{epochs} - Start training =======================")
            self._epoch = epoch
            self._one_step(X_train, y_train, batch_size=batch_size, training=True, verbose=verbose, device=device)

            # self.loss_meter.reset()
            # self.acc_meter.reset()

            loss, acc = self._one_step(X_val, y_val, batch_size=len(X_val), training=False, verbose=False, device=device)
            if verbose:
                print(f">>>>>>>>>>> Epoch loss: {loss:.4f} - Epoch acc: {acc:.4f}")
            toc = time.time()
            if verbose:
                print(f"Time cost: {toc-tic:.2f} seconds")

    def _one_step(self, X_data, y_data, batch_size=32, training=True, verbose=True, device='cpu'):
        if training: # shuffle
            idx = np.random.permutation(len(X_data))
            X = X_data[idx]
            y = y_data[idx]
        else:
            X = X_data
            y = y_data
        
        loss_sum, acc_sum, sample_num = 0., 0, 0
        for start_idx in range(0, len(X), batch_size):
            Xb = Tensor(X[start_idx:start_idx+batch_size])
            yb = y[start_idx:start_idx+batch_size]

            y_hat = self.model(Xb)
            if yb.ndim > 1:
                y_true = yb.argmax(axis=1)
            else:
                y_true = yb

            loss = self.loss_fn(y_hat, y_true)

            if y_hat.ndim > 1:
                y_hat = y_hat.argmax(axis=1)
            else:
                y_hat = y_hat

            if training:
                self.model.cleargrads()
                loss.backward()
                self.optimizer.update()
            
            loss_sum += loss.data * len(Xb)
            acc_sum += (y_hat == y_true).sum()
            sample_num += len(Xb)

            # self.loss_meter.update(loss.data)
            # self.acc_meter.update(acc_sum / sample_num)
            
            if verbose:
                progress_bar(start_idx + len(Xb), len(X), self._epoch, loss.data, acc_sum / sample_num)
        if verbose:
            sys.stdout.write('\n')

        return loss_sum / sample_num, acc_sum / sample_num 

class Evaluator:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        # self.loss_meter = AverageMeter('Loss')
        # self.acc_meter = AverageMeter('Acc')
        # self.time_meter = TimeMeter()

    def evaluate(self, data_loader, batch_size=32, verbose=True, device='cpu'):
        loss_sum, acc_sum, sample_num = 0., 0, 0
        if verbose:
            print(f"======================= Start evaluation =======================")

        # self.model.to(device)
        X, y = data_loader.get_data()
        for start_idx in range(0, len(X), batch_size):
            Xb = Tensor(X[start_idx:start_idx+batch_size])
            yb = y[start_idx:start_idx+batch_size]

            with usingConfig('train', False):
                y_hat = self.model(Xb)
                
                if yb.ndim > 1:
                    y_true = yb.argmax(axis=1)
                else:
                    y_true = yb
                loss = self.loss_fn(y_hat, y_true) 
            
            if y_hat.ndim > 1:
                y_hat = y_hat.data.argmax(axis=1)
            else:
                y_hat = y_hat.data
            loss_sum += loss.data * len(Xb)
            acc_sum += (y_hat == y_true).sum()
            sample_num += len(Xb)

            # self.loss_meter.update(loss.data)
            # self.acc_meter.update(acc_sum / sample_num)
        
        return loss_sum / sample_num, acc_sum / sample_num

        
'''
