import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# === 数据生成函数 ===
def generate_data(num_samples, image_size=128):
    """
    生成随机形状（圆形或正方形）的二值图像数据集。
    每个样本包含一个形状，输入图像和对应的掩码图像相同。

    参数:
    - num_samples: 样本数量
    - image_size: 图像尺寸（宽和高）

    返回:
    - X: 输入图像数据集
    - Y: 掩码图像数据集
    """
    X = np.zeros((num_samples, image_size, image_size, 1), dtype=np.float32)
    Y = np.zeros((num_samples, image_size, image_size, 1), dtype=np.float32)

    for i in range(num_samples):
        # 创建随机形状（圆形或正方形）
        shape_type = np.random.choice(['circle', 'square'])
        x, y = np.random.randint(20, image_size - 20, size=2)
        size = np.random.randint(10, 30)

        if shape_type == 'circle':
            y_grid, x_grid = np.ogrid[:image_size, :image_size]
            mask = (x_grid - x)**2 + (y_grid - y)**2 <= size**2
        elif shape_type == 'square':
            mask = np.zeros((image_size, image_size), dtype=bool)
            mask[y-size:y+size, x-size:x+size] = True

        # 将形状添加到输入图像 (X) 和掩码图像 (Y)
        X[i, ..., 0] = mask.astype(float)
        Y[i, ..., 0] = mask.astype(float)

    return X, Y

# 生成数据集
num_samples = 1000
image_size = 128
X, Y = generate_data(num_samples, image_size=image_size)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    # 编码器
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # 解码器
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    return models.Model(inputs, outputs)

# 构建并编译模型
input_shape = (image_size, image_size, 1)
model = build_unet(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 训练模型
history = model.fit(
    X_train, Y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=16
)

# 在测试集上进行预测
Y_pred = model.predict(X_test)

# 可视化结果
def plot_sample(X, Y_true, Y_pred, index):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("输入图像")
    plt.imshow(X[index, ..., 0], cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("真实掩码")
    plt.imshow(Y_true[index, ..., 0], cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("预测掩码")
    plt.imshow(Y_pred[index, ..., 0], cmap='gray')

    plt.show()

# 显示2个随机示例
for i in np.random.choice(len(X_test), 2):
    plot_sample(X_test, Y_test, Y_pred, i)