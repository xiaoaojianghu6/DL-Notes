# 项目一：LeNet-5 复现

## 1. 项目背景

LeNet-5由大师杨立昆在1998年提出。通过在标准的MNIST手写数字数据集上从零开始构建、训练和评估LeNet-5，可以熟悉CNN的核心组件（卷积、池化、全连接层）及其协同工作的方式。

## 2. 核心实现

### 2.1. LeNet-5 模型结构

我们严格按照LeNet-5的经典结构进行搭建。模型包含两个卷积层、两个最大池化层和三个全连接层，最后通过激活函数（ReLU）和输出层完成分类任务。

**可复用的模型定义模块** (`CNN_LeNet.py`):
```python
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层1: 输入1通道，输出6通道，5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 卷积层2: 输入6通道，输出16通道，5x5卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 全连接层1: 输入16*4*4=256个节点，输出120个节点
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 全连接层2: 输入120个节点，输出84个节点
        self.fc2 = nn.Linear(120, 84)
        # 全连接层3: 输入84个节点，输出10个节点 (对应0-9十个数字)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积 -> ReLU -> 池化
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        # 卷积 -> ReLU -> 池化
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        # 展平以输入全连接层
        x = x.view(-1, 16 * 4 * 4)
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 2.2. 训练与评估

使用标准的交叉熵损失函数 (`nn.CrossEntropyLoss`) 和Adam优化器 (`optim.Adam`) 对模型进行训练。经过5个epoch的训练，模型在测试集上达到了 **98.41%** 的高准确率，成功复现了LeNet-5在MNIST任务上的优异表现。

## 3\. 结果可视化与分析


### 3.1. 预测结果抽样展示

我们随机抽取了部分测试集样本，将模型的预测结果与真实标签进行了对比。如下图所示，模型能够准确识别各种手写数字，即使是一些书写较为潦草的样本。
![图片](/projects/LeNet-5/1.png) 


  ### 3.2. 混淆矩阵 (Confusion Matrix)
从下方的混淆矩阵可以看出，绝大多数样本都被正确分类（集中在对角线上）。少量错误主要发生在形态相似的数字之间，例如将“4”误判为“9”，或将“7”误判为“2”，完全符合直觉。

![图片](/projects/LeNet-5/2.png) 

 ## 4. 可复用代码文件

  - **完整实现**: [`CNN_LeNet.py`](/projects/LeNet-5/CNN_LeNet.py)

