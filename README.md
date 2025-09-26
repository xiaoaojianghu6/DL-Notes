# DL-Notes
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

本仓库主要记录我在深度学习的学习心得、模型实现与项目实践。

## 仓库结构

-   **/cv-basics**: 存放计算机视觉基础，如图像预处理的通用代码范式。
-   **/projects**: 包含具体的深度学习项目实践，涵盖经典模型的复现与应用。
-   **/cs231n-assignments**: (规划中) 整理后将在此处记录CS231n课程的作业解答尝试与笔记。

---

## 可复用模块

### 1. 图像预处理

* **[图像预处理通用范式](./cv-basics/image-preprocessing/README.md)**
    * **简介**: 总结使用`OpenCV`库进行的一系列常用图像预处理操作，包括尺寸统一、灰度化、去噪滤波、边缘检测和对比度增强。
---

## 项目实践 (更新中)

* ### [项目一：卷积神经网络 LeNet-5 复现](./projects/LeNet-5/README.md)
    * **简介**: 在MNIST手写数字数据集上，从零开始完整复现了经典的LeNet-5卷积神经网络模型。
    * **技术栈**: `PyTorch`, `Torchvision`, `Scikit-learn`
    * **核心知识**: 卷积层、池化层、全连接层、模型训练与评估、混淆矩阵可视化。

* ### [项目二：图像分割 U-Net 模型实践](./projects/U-Net/README.md)
    * **简介**: 实践了U-Net这一经典的图像分割模型。通过生成简单的几何形状数据集，训练模型学习从输入图像中精确地分割出目标区域（掩码）。
    * **技术栈**: `TensorFlow`, `Keras`, `Matplotlib`
    * **核心知识**: 编码器-解码器结构、跳跃连接（Skip Connection）、卷积与反卷积。

* ### [项目三：CNN与RNN在ECG信号分类中的应用对比](./projects/ECG-CNN-vs-RNN/README.md)
    * **简介**: 对比CNN和RNN在处理ECG时间序列数据时的性能差异。
    * **核心发现**: RNN凭借其对序列信息的记忆能力，在需要结合上下文模式进行判断的任务中（如房颤识别）可能更具优势。而CNN则更擅长从信号片段中提取局部“形态”特征。

---

## 许可证 (License)

本仓库采用 [MIT 许可证](./LICENSE)。
