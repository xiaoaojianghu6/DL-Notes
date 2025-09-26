# 项目二：图像分割 U-Net 模型实践

## 1. 项目背景

U-Net是生物医学图像分割的经典模型。通过一个编码器（Encoder）路径来捕捉图像的上下文特征，并通过一个对称的解码器（Decoder）路径来实现精确定位，最终输出与原图大小一致的分割掩码（Mask）。

因为是简单实践，本次实验数据通过程序**动态生成**了包含简单几何形状（圆形和正方形）的二值图像，任务是让U-Net模型学习如何从输入图像中完美地分割出这些形状。

## 2. 核心实现

这里使用`TensorFlow`和`Keras`库实现从数据生成、U-Net模型构建到训练和结果可视化的全过程。
题外话，非常不幸的是，笔者在这里整理笔记时，`tensorflow`貌似已经行将就木，在这里表示遗憾和惋惜。但对这种基础内容还是够用且便捷的。

### 2.1. 数据生成

`generate_data`函数会在一个128x128的黑色背景上，随机生成不同位置和大小的白色圆形或正方形。

**可复用的数据生成模块**:
```python
import numpy as np

def generate_data(num_samples, image_size=128):
    """生成随机形状（圆形或正方形）的二值图像数据集。"""
    X = np.zeros((num_samples, image_size, image_size, 1), dtype=np.float32)
    Y = np.zeros((num_samples, image_size, image_size, 1), dtype=np.float32)

    for i in range(num_samples):
        shape_type = np.random.choice(['circle', 'square'])
        # ... (随机生成形状的位置和大小) ...
        # ... (绘制形状到X和Y) ...
    return X, Y

# 生成1000个样本用于训练和测试
X, Y = generate_data(num_samples=1000, image_size=128)
```

### 2.2. U-Net 模型架构

U-Net的核心在于其对称的编码器-解码器结构以及**跳跃连接**机制。

1.  **编码器（下采样路径）**: 通过一系列的卷积（Conv2D）和最大池化（MaxPooling2D）层，逐步提取图像的深层语义特征，同时减小特征图的尺寸。
2.  **解码器（上采样路径）**: 使用反卷积层（Conv2DTranspose）将特征图逐步放大回原始尺寸。
3.  **跳跃连接**: 将编码器中对应层级的特征图（保留了高分辨率的细节信息）与解码器上采样后的特征图进行拼接（`concatenate`）。
   这使得解码器在恢复图像细节时，能够同时利用深层的语义信息和浅层的细节信息，从而实现非常精确的分割。

**模型构建核心代码** (`u-net.py`):

```python
from tensorflow.keras import layers, models

def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    # --- 编码器 ---
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2) # 瓶颈层

    # --- 解码器 ---
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2]) # 跳跃连接
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1]) # 跳跃连接
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    
    # 输出层，使用sigmoid激活函数进行二值分割
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    return models.Model(inputs, outputs)
```

## 3\. 实验结果与分析

使用`adam`优化器和`binary_crossentropy`损失函数进行训练。经过10个epoch的训练。
从下面的预测结果可以看出，模型生成的预测掩码(第三组)与真实掩码(第二组)几乎完全一致。中文显示稍稍有点问题请忽略。
**预测结果可视化**:
![图片](/projects/U-Net/1.png) 


  ## 4. 可复用代码文件

  - **完整实现**: [`u-net.py`](/projects/U-Net/u-net.py)

