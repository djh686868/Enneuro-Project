import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import zoom
from ..train.meters import AverageMeter, TimeMeter
from ..train.metrics import confusion_matrix as _compute_cm

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
    
    @staticmethod
    def _to_numpy_flat(arr):
        """将 Tensor / cupy / list 统一转为一维 numpy int64 数组。"""
        if type(arr).__name__ == 'Tensor' and hasattr(arr, 'data'):
            arr = arr.data
        try:
            import cupy as cp
            if isinstance(arr, cp.ndarray):
                arr = cp.asnumpy(arr)
        except ImportError:
            pass
        return np.asarray(arr, dtype=np.int64).ravel()

    def update_predictions(self, y_true, y_pred):
        """更新预测结果，用于绘制混淆矩阵"""
        self.y_true.extend(self._to_numpy_flat(y_true).tolist())
        self.y_pred.extend(self._to_numpy_flat(y_pred).tolist())
    
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
            y_true_np = np.array(self.y_true, dtype=np.int64)
            y_pred_np = np.array(self.y_pred, dtype=np.int64)
            cm = _compute_cm(y_pred_np, y_true_np, num_classes=self.num_classes)
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


class GradCAMVisualizer:
    """
    Grad-CAM 可视化器
    
    提供 Grad-CAM、Guided Backpropagation 和 Guided Grad-CAM 的可视化功能。
    
    参考论文:
    - Grad-CAM: https://arxiv.org/abs/1610.02391
    - Guided Backpropagation: https://arxiv.org/abs/1312.6034
    
    使用示例:
    >>> visualizer = GradCAMVisualizer(model, target_layer)
    >>> visualizer.visualize(input_tensor, class_idx=0, save_path='gradcam.png')
    """

    def __init__(self, model, target_layer=None):
        """
        初始化 Grad-CAM 可视化器
        
        Args:
            model: 神经网络模型
            target_layer: Grad-CAM 的目标层，默认为最后一个卷积层
        """
        self.model = model
        
        # 延迟导入以避免循环依赖
        from ..explainability import GradCAM, GuidedBackpropagation, GuidedGradCAM, suggest_target_layer
        
        self.target_layer = target_layer if target_layer else suggest_target_layer(model)
        
        if self.target_layer is None:
            raise ValueError("无法找到合适的目标层，请手动指定 target_layer")
        
        self.gradcam = GradCAM(model, self.target_layer)
        self.guided_bp = GuidedBackpropagation(model)
        self.guided_gradcam = GuidedGradCAM(model, self.target_layer)

    def visualize(self, input_tensor, class_idx=None, save_path=None, show=True):
        """
        可视化单个图像的 Grad-CAM 结果
        
        Args:
            input_tensor: 输入张量 (N, C, H, W)
            class_idx: 目标类别索引，默认为预测类别
            save_path: 保存路径
            show: 是否显示
        
        Returns:
            可视化图像（如果 show=True 则显示）
        """
        # 获取预测结果
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(np.argmax(output.data, axis=1)[0])
        pred_label = class_idx
        
        # 生成可视化结果
        heatmap = self.gradcam.generate(input_tensor, class_idx)
        saliency = self.guided_bp.generate(input_tensor, class_idx)
        guided_gc = self.guided_gradcam.generate(input_tensor, class_idx)
        
        # 准备原始图像
        original_img = input_tensor.data[0].squeeze()
        if original_img.ndim == 3:
            original_img = original_img.transpose(1, 2, 0)
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)
        
        # 调整热力图大小以匹配原始图像
        if heatmap.shape != original_img.shape[:2]:
            zoom_factor = (original_img.shape[0] / heatmap.shape[0], original_img.shape[1] / heatmap.shape[1])
            heatmap_resized = zoom(heatmap, zoom_factor, order=1)
        else:
            heatmap_resized = heatmap
        
        # 创建可视化图像
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 1. 原始图像
        axes[0].imshow(original_img, cmap='gray' if original_img.ndim == 2 else None)
        axes[0].set_title(f'Original Image\n(Predicted: {pred_label})', fontsize=14)
        axes[0].axis('off')
        
        # 2. Grad-CAM
        im1 = axes[1].imshow(heatmap_resized, cmap='jet', interpolation='bilinear', vmin=0, vmax=1)
        axes[1].set_title('Grad-CAM', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. Guided Backpropagation
        saliency_vis = saliency.squeeze() if saliency.ndim == 3 else saliency
        im2 = axes[2].imshow(saliency_vis, cmap='gray', interpolation='bilinear')
        axes[2].set_title('Guided Backpropagation', fontsize=14)
        axes[2].axis('off')
        
        # 4. Guided Grad-CAM
        guided_vis = guided_gc.squeeze() if guided_gc.ndim == 3 else guided_gc
        im3 = axes[3].imshow(guided_vis, cmap='gray', interpolation='bilinear')
        axes[3].set_title('Guided Grad-CAM', fontsize=14)
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()
        
        return {
            'heatmap': heatmap,
            'saliency': saliency,
            'guided_gradcam': guided_gc,
            'predicted_class': pred_label
        }

    def visualize_comparison(self, input_tensors, class_indices=None, save_path=None, show=True):
        """
        可视化多个图像的 Grad-CAM 结果对比
        
        Args:
            input_tensors: 输入张量列表
            class_indices: 类别索引列表
            save_path: 保存路径
            show: 是否显示
        """
        num_samples = len(input_tensors)
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
        
        for i, input_tensor in enumerate(input_tensors):
            class_idx = class_indices[i] if class_indices else None
            
            result = self.visualize_single(input_tensor, class_idx, show=False)
            
            original_img = result['original']
            heatmap = result['heatmap']
            saliency = result['saliency']
            guided_gc = result['guided_gradcam']
            pred_label = result['predicted_class']
            
            row = axes[i] if num_samples > 1 else axes
            
            row[0].imshow(original_img, cmap='gray' if original_img.ndim == 2 else None)
            row[0].set_title(f'Image {i+1}\n(Pred: {pred_label})', fontsize=12)
            row[0].axis('off')
            
            row[1].imshow(heatmap, cmap='jet', interpolation='bilinear', vmin=0, vmax=1)
            row[1].set_title('Grad-CAM', fontsize=12)
            row[1].axis('off')
            
            row[2].imshow(saliency.squeeze() if saliency.ndim == 3 else saliency, cmap='gray')
            row[2].set_title('Guided BP', fontsize=12)
            row[2].axis('off')
            
            row[3].imshow(guided_gc.squeeze() if guided_gc.ndim == 3 else guided_gc, cmap='gray')
            row[3].set_title('Guided Grad-CAM', fontsize=12)
            row[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()

    def visualize_single(self, input_tensor, class_idx=None, show=True):
        """
        可视化单个图像（内部方法，返回数据而不显示）
        
        Args:
            input_tensor: 输入张量
            class_idx: 类别索引
            show: 是否显示
        
        Returns:
            包含可视化数据的字典
        """
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(np.argmax(output.data, axis=1)[0])
        
        heatmap = self.gradcam.generate(input_tensor, class_idx)
        saliency = self.guided_bp.generate(input_tensor, class_idx)
        guided_gc = self.guided_gradcam.generate(input_tensor, class_idx)
        
        original_img = input_tensor.data[0].squeeze()
        if original_img.ndim == 3:
            original_img = original_img.transpose(1, 2, 0)
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)
        
        if heatmap.shape != original_img.shape[:2]:
            zoom_factor = (original_img.shape[0] / heatmap.shape[0], original_img.shape[1] / heatmap.shape[1])
            heatmap = zoom(heatmap, zoom_factor, order=1)
        
        return {
            'original': original_img,
            'heatmap': heatmap,
            'saliency': saliency,
            'guided_gradcam': guided_gc,
            'predicted_class': class_idx
        }

    def visualize_layer_comparison(self, input_tensor, layers=None, class_idx=None, save_path=None, show=True):
        """
        可视化不同层的 Grad-CAM 结果对比
        
        Args:
            input_tensor: 输入张量
            layers: 要对比的层列表，默认为所有卷积层
            class_idx: 类别索引
            save_path: 保存路径
            show: 是否显示
        """
        from ..explainability import get_all_conv_layers, GradCAM
        
        if layers is None:
            layers = [layer for _, layer in get_all_conv_layers(self.model)]
        
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(np.argmax(output.data, axis=1)[0])
        
        num_layers = len(layers)
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
        
        for i, layer in enumerate(layers):
            gradcam = GradCAM(self.model, layer)
            heatmap = gradcam.generate(input_tensor, class_idx)
            
            im = axes[i].imshow(heatmap, cmap='jet', interpolation='bilinear', vmin=0, vmax=1)
            axes[i].set_title(f'{type(layer).__name__}\nShape: {heatmap.shape}', fontsize=12)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Grad-CAM on Different Layers (Class: {class_idx})', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()

    def visualize_class_comparison(self, input_tensor, classes=None, save_path=None, show=True):
        """
        可视化同一图像对不同类别的 Grad-CAM 结果
        
        Args:
            input_tensor: 输入张量
            classes: 要对比的类别列表，默认为 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            save_path: 保存路径
            show: 是否显示
        """
        if classes is None:
            classes = list(range(10))
        
        num_classes = len(classes)
        cols = min(num_classes, 5)
        rows = (num_classes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        
        for i, class_idx in enumerate(classes):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            heatmap = self.gradcam.generate(input_tensor, class_idx)
            
            im = ax.imshow(heatmap, cmap='jet', interpolation='bilinear', vmin=0, vmax=1)
            ax.set_title(f'Class {class_idx}', fontsize=12)
            ax.axis('off')
        
        plt.suptitle('Grad-CAM for Different Classes', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()

    def generate_overlay(self, input_tensor, class_idx=None, alpha=0.5, save_path=None, show=True):
        """
        生成 Grad-CAM 热力图叠加在原始图像上的可视化
        
        Args:
            input_tensor: 输入张量
            class_idx: 类别索引
            alpha: 热力图透明度
            save_path: 保存路径
            show: 是否显示
        
        Returns:
            叠加后的图像
        """
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(np.argmax(output.data, axis=1)[0])
        
        heatmap = self.gradcam.generate(input_tensor, class_idx)
        
        original_img = input_tensor.data[0].squeeze()
        if original_img.ndim == 3:
            original_img = original_img.transpose(1, 2, 0)
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)
        
        if heatmap.shape != original_img.shape[:2]:
            zoom_factor = (original_img.shape[0] / heatmap.shape[0], original_img.shape[1] / heatmap.shape[1])
            heatmap = zoom(heatmap, zoom_factor, order=1)
        
        heatmap_colored = np.array(plt.cm.jet(heatmap))[:, :, :3]
        
        if original_img.ndim == 2:
            overlay = alpha * original_img[:, :, np.newaxis] + (1 - alpha) * heatmap_colored
        else:
            overlay = alpha * original_img + (1 - alpha) * heatmap_colored
        
        overlay = np.clip(overlay, 0, 1)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(overlay)
        ax.set_title(f'Grad-CAM Overlay (Class: {class_idx})', fontsize=14)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()
        
        return overlay
