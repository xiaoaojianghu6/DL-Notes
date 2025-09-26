# 项目三：ECG信号分类 - CNN vs. RNN 应用对比

## 1. 项目背景与数据集

数据集使用国际通用的 **MIT-BIH 心律失常数据库**，包含多种心搏类型的标注。

目标是根据ECG信号中的R波（心搏的关键特征点）周围的波形对心搏类型（如正常搏动N、房性早搏A、室性早搏V等）进行分类。

### 核心问题

CNN通常被认为是图像处理的利器，擅长提取空间局部特征；
而RNN则为处理序列数据而生，能够捕捉时间上的依赖关系。
所以，在处理一维的ECG信号时，这两种模型各有什么优势和劣势？

-   **CNN**: 将R波周围的信号段视为一个一维“图像”，通过卷积核来学习和识别不同心搏类型的“波形形态”特征。
-   **RNN**: 将信号点按时间顺序输入，通过其内部的“记忆”机制（隐藏状态），结合上下文信息来判断当前心搏的类型。

## 2. 核心实现

### 2.1. 可复用的数据预处理范式

这是一个处理ECG信号的通用范式，可以被复用于多种分析任务。

1.  **数据加载**: 使用`wfdb`库读取MIT-BIH数据库中的`.dat`（信号）和`.atr`（注释）文件。
2.  **信号去噪**: 采用**小波变换（Wavelet Transform）**对原始信号进行去噪。小波变换能有效地分离信号和噪声，保留心搏的关键特征。
3.  **数据切片**: 以R波位置为中心，向前截取100个数据点，向后截取200个数据点，形成一个长度为300的一维向量，代表一次完整的心搏。

**可复用的数据处理模块** (`CNN_pm.ipynb`, `RNN_pm.ipynb`):
```python
import wfdb
import pywt
import numpy as np

# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    # ... (阈值去噪等) ...
    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

# 读取心电数据和对应标签,并进行切片
def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']
    # 读取心电数据记录
    record = wfdb.rdrecord(f'/path/to/mit-bih/{number}', channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)
    
    # 获取R波位置和标签
    annotation = wfdb.rdann(f'/path/to/mit-bih/{number}', 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol
    
    # ...
    # 在R波前后截取长度为300的数据点
    x_train = rdata[Rlocation[i] - 100:Rlocation[i] + 200]
    X_data.append(x_train)
```

### 2.2. 模型架构

我们分别使用`TensorFlow`和`PyTorch`不同框架实现了CNN和RNN模型。

#### **1D-CNN 模型** (`TensorFlow/Keras`)

模型包含多个一维卷积层（`Conv1D`）和池化层（`MaxPool1D`/`AvgPool1D`），用于逐层提取ECG波形的局部形态特征，最后通过全连接层进行分类。

```python
import tensorflow as tf

def buildModel():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(300, 1)),
        # 第一个卷积-池化块
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='same', activation='tanh'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        # ... (更多卷积-池化块) ...
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return newModel
```

#### **RNN 模型** (`PyTorch`)

模型使用了一个包含3个隐藏层的RNN网络。输入数据被视为一个长度为300的时间序列，RNN通过处理整个序列来捕捉心搏的动态变化趋势。

```python
import torch.nn as nn

class RnnModel(nn.Module):
    def __init__(self):
        super(RnnModel, self).__init__()
        # 输入维度300, 隐藏层维度50, 网络层数3
        self.rnn = nn.RNN(300, 50, 3, nonlinearity='tanh')
        self.linear = nn.Linear(50, 5)

    def forward(self, x):
        r_out, h_state = self.rnn(x)
        # 取最后一个时间步的输出进行分类
        output = self.linear(r_out[:,-1,:])
        return output
```

## 3\. 实验结果与分析（具体见代码）

  - **CNN**: 经过30个epoch的训练，在测试集上达到了99.46%的准确率，展现了其强大的局部特征提取能力。
  - **RNN**: 经过10个epoch的训练，在测试集上达到了96.52%的准确率，同样表现不俗。

**结论分析**:

  - 对于**单次心搏形态**的分类任务，CNN表现出了微弱的优势。这可能是因为单次心搏的类别判断主要依赖于其P-QRS-T波的特定形态，是CNN卷积核所擅长捕捉的“局部模式”。
  - 然而，如我在学习报告中分析的，对于需要**结合上下文、观察多个R波间期变化**才能判断的复杂心律失常（如房颤），RNN凭借其对序列信息的“记忆”能力，理论上会更具优势。

## 4\. 可复用代码文件

  - **CNN实现 (TensorFlow)**: [`CNN_pm.ipynb`](/projects/ECG-CNN-vs-RNN/CNN_pm.ipynb)
  - **RNN实现 (PyTorch)**: [`RNN_pm.ipynb`](/projects/ECG-CNN-vs-RNN/RNN_pm.ipynb)

