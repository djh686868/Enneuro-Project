VISUAL_CONFIG={
    "ENABLE_ALL_LAYERS": False
}

# eneuro/utils/visualize_tools.py
import cv2



def visualize_model_first_batch(model, data_loader):
    """
    通用可视化工具：可视化任意模型在数据加载器第一个批次的特征图
    适配所有自定义模型（LeNet/CNN/ResNet等），无需修改模型代码

    Args:
        model: 任意神经网络模型（需支持 __call__ 前向传播）
        data_loader: 数据加载器（需支持 __iter__ 迭代，如 SimpleDataLoader）

    Returns:
        None

    Raises:
        ValueError: 数据加载器为空或不支持迭代
        Exception: 前向传播失败时抛出原始异常（便于调试）
    """
    # ========== 1. 前置校验（保证鲁棒性） ==========
    # 校验数据加载器是否支持迭代
    if not hasattr(data_loader, '__iter__'):
        raise ValueError(f"数据加载器 {type(data_loader)} 不支持迭代，请实现 __iter__ 方法")

    # ========== 2. 准备可视化 ==========
    print("\n=== 开始可视化【第一个批次】特征图 ===")
    # 临时开启可视化开关（保存原始状态，避免覆盖）
    original_switch_state = VISUAL_CONFIG.get("ENABLE_ALL_LAYERS", False)
    VISUAL_CONFIG["ENABLE_ALL_LAYERS"] = True

    try:
        # ========== 3. 仅提取第一个批次的输入数据 ==========
        first_batch = next(iter(data_loader))  # 取第一个批次
        # 兼容数据加载器返回格式：(Xb, yb) 或 仅 Xb
        if isinstance(first_batch, (tuple, list)):
            Xb, _ = first_batch  # 只保留输入数据，丢弃标签
        else:
            Xb = first_batch  # 适配仅返回输入的情况

        # ========== 4. 执行前向传播（触发特征图可视化） ==========
        # 核心：仅输入第一个批次数据，不计算任何指标，纯可视化
        _ = model(Xb)

    except StopIteration:
        raise ValueError("数据加载器为空，无任何批次数据可可视化")
    except Exception as e:
        raise Exception(f"可视化前向传播失败：{str(e)}")
    finally:
        # ========== 5. 恢复状态 + 清理资源（关键） ==========
        # 强制恢复可视化开关原始状态，避免影响后续逻辑
        VISUAL_CONFIG["ENABLE_ALL_LAYERS"] = original_switch_state
        # 清理所有OpenCV窗口，避免资源泄漏
        cv2.destroyAllWindows()
        print("=== 第一个批次特征图可视化完成 ===\n")