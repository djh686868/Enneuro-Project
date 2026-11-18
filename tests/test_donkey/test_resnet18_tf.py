import os
import json
import time
import psutil
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf


def progress_bar(current, total, epoch, loss, acc):
    bar_length = 30
    progress = current / total
    arrow = '=' * int(round(progress * bar_length))
    spaces = '-' * (bar_length - len(arrow))
    print(f'\rEpoch [{epoch+1}] [{arrow}{spaces}] {current}/{total} - loss: {loss:.4f} - acc: {acc:.4f}', end='')


def count_params(model):
    return model.count_params()


def build_resnet18(num_classes=20):
    inputs = tf.keras.Input(shape=(120, 160, 3))
    
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    def block(x, filters, strides=1, downsample=None):
        residual = x
        x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        if downsample is not None:
            residual = downsample(residual)
        
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    filters = 64
    for _ in range(2):
        x = block(x, filters)
    
    filters = 128
    x = block(x, filters, strides=2, downsample=tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size=1, strides=2, use_bias=False),
        tf.keras.layers.BatchNormalization()
    ]))
    x = block(x, filters)
    
    filters = 256
    x = block(x, filters, strides=2, downsample=tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size=1, strides=2, use_bias=False),
        tf.keras.layers.BatchNormalization()
    ]))
    x = block(x, filters)
    
    filters = 512
    x = block(x, filters, strides=2, downsample=tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size=1, strides=2, use_bias=False),
        tf.keras.layers.BatchNormalization()
    ]))
    x = block(x, filters)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            parts = filename.split('_')
            if len(parts) >= 2:
                angle = float(parts[-1].split('.')[0])
                with Image.open(img_path) as pil_img:
                    pil_img = pil_img.convert('RGB')
                    img = np.array(pil_img, dtype=np.float32) / 255.0
                images.append(img)
                labels.append(int((angle + 1) * 10))
    
    return np.array(images), np.array(labels)


def train():
    print("GPU available:", bool(tf.config.list_physical_devices('GPU')))
    
    batch_size = 16
    num_workers = 2
    
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir / 'data'
    
    images, labels = load_images_from_folder(str(data_path))
    num_samples = len(images)
    
    tf_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    tf_dataset = tf_dataset.shuffle(num_samples)
    tf_dataset = tf_dataset.batch(batch_size)
    
    model = build_resnet18(num_classes=20)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    total_params = count_params(model)
    print(f"Model parameters: {total_params:,}")
    
    mem_peak = 0
    gpu_mem_history = []
    
    num_epoch = 1
    
    for epoch in range(num_epoch):
        tic = time.time()
        loss_sum, acc_sum, sample_num = 0., 0, 0
        
        for batch_idx, (images_batch, labels_batch) in enumerate(tf_dataset):
            if images_batch.shape[0] != batch_size:
                break
            
            with tf.GradientTape() as tape:
                y_pre = model(images_batch, training=True)
                loss = loss_fn(labels_batch, y_pre)
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            pred_classes = tf.argmax(y_pre, axis=1)
            batch_acc = tf.reduce_mean(tf.cast(tf.equal(pred_classes, labels_batch), tf.float32)).numpy()
            
            loss_sum += loss.numpy() * len(images_batch)
            if not np.isnan(batch_acc):
                acc_sum += batch_acc * len(images_batch)
            sample_num += len(images_batch)
            
            display_acc = (acc_sum / sample_num) if sample_num > 0 and not np.isnan(acc_sum) else np.nan
            progress_bar(batch_idx * batch_size + len(images_batch), num_samples, epoch, loss.numpy(), display_acc)
            
            current_mem = psutil.Process().memory_info().rss / 1024**2
            mem_peak = max(mem_peak, current_mem)
            
            if tf.config.list_physical_devices('GPU'):
                try:
                    gpus = tf.config.list_logical_devices('GPU')
                    for gpu in gpus:
                        mem_info = tf.config.experimental.get_memory_info(gpu.name)
                        gpu_mem_history.append(mem_info['current'] / 1024**2)
                        break
                except Exception:
                    pass
        
        toc = time.time()
        duration = toc - tic
        print(f"\nEpoch completed in {duration:.4f}s\n")
    
    gpu_peak = max(gpu_mem_history) if gpu_mem_history else 0
    gpu_avg = np.mean(gpu_mem_history) if gpu_mem_history else 0
    
    result = {
        'framework': 'TensorFlow',
        'duration': float(duration),
        'params': int(total_params),
        'mem_peak': float(mem_peak),
        'gpu_peak': float(gpu_peak),
        'gpu_avg': float(gpu_avg),
        'num_workers': int(num_workers),
        'batch_size': int(batch_size),
        'num_samples': int(num_samples)
    }
    
    results_dir = current_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'benchmark_tf.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print("\nBenchmark results:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result


if __name__ == '__main__':
    train()
