import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from ..train.meters import AverageMeter, TimeMeter

class Visualizer:
    """用于可视化训练过程中的准确率曲线、损失曲线、时间消耗曲线和混淆矩阵"""
    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        
        # 训练指标
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.epoch_times = []
        
        # 用于收集预测结果和真实标签，以便绘制混淆矩阵
        self.y_true = []
        self.y_pred = []
        
        # 使用meter中的计数器
        self.train_loss_meter = AverageMeter('Train Loss')
        self.train_acc_meter = AverageMeter('Train Acc')
        self.val_loss_meter = AverageMeter('Val Loss')
        self.val_acc_meter = AverageMeter('Val Acc')
        self.time_meter = TimeMeter()
    
    def reset(self):
        """重置所有指标和计数器"""
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.epoch_times = []
        self.y_true = []
        self.y_pred = []
        
        self.train_loss_meter.reset()
        self.train_acc_meter.reset()
        self.val_loss_meter.reset()
        self.val_acc_meter.reset()
        self.time_meter.reset()
    
    def update_train(self, loss, acc, batch_size=1):
        """更新训练指标"""
        self.train_loss_meter.update(loss, batch_size)
        self.train_acc_meter.update(acc, batch_size)
    
    def update_val(self, loss, acc, batch_size=1):
        """更新验证指标"""
        self.val_loss_meter.update(loss, batch_size)
        self.val_acc_meter.update(acc, batch_size)
    
    def update_epoch(self, epoch_time):
        """更新 epoch 时间"""
        self.epoch_times.append(epoch_time)
        self.train_loss.append(self.train_loss_meter.avg)
        self.train_acc.append(self.train_acc_meter.avg)
        self.val_loss.append(self.val_loss_meter.avg)
        self.val_acc.append(self.val_acc_meter.avg)
        
        # 重置训练和验证指标计数器，为下一个 epoch 做准备
        self.train_loss_meter.reset()
        self.train_acc_meter.reset()
        self.val_loss_meter.reset()
        self.val_acc_meter.reset()
    
    def update_predictions(self, y_true, y_pred):
        """更新预测结果，用于绘制混淆矩阵"""
        # 确保输入是 numpy 数组
        if not isinstance(y_true, np.ndarray):
            y_true = y_true.data if hasattr(y_true, 'data') else np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.data if hasattr(y_pred, 'data') else np.array(y_pred)
        
        self.y_true.extend(y_true.flatten())
        self.y_pred.extend(y_pred.flatten())
    
    def plot_all(self, save_path=None, show=True):
        """一次性绘制所有曲线和混淆矩阵"""
        # 图像大小缩小到原来的3/4，从(15, 12)变为(11.25, 9)
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
        fig.suptitle('Training Visualization', fontsize=16)
        
        # 1. 绘制损失曲线
        axes[0, 0].plot(self.train_loss, label='Train Loss')
        axes[0, 0].plot(self.val_loss, label='Val Loss')
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 绘制准确率曲线（包括训练准确率和验证准确率）
        axes[0, 1].plot(self.train_acc, label='Train Acc')
        axes[0, 1].plot(self.val_acc, label='Val Acc')
        axes[0, 1].set_title('Accuracy Curve')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. 绘制时间消耗曲线
        axes[1, 0].plot(self.epoch_times)
        axes[1, 0].set_title('Time Consumption per Epoch')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True)
        
        # 4. 绘制混淆矩阵
        if len(self.y_true) > 0 and len(self.y_pred) > 0:
            cm = confusion_matrix(self.y_true, self.y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_title('Confusion Matrix')
            axes[1, 1].set_xlabel('Predicted Label')
            axes[1, 1].set_ylabel('True Label')
        else:
            axes[1, 1].text(0.5, 0.5, 'No prediction data available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()
