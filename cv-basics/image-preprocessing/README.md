# 图像预处理
## 1. 流程概述

一个标准的图像预处理流水线（pipeline）通常包括以下步骤：

**尺寸统一 -> 灰度化 -> 去噪/平滑 -> 边缘检测 -> 对比度增强**

这些步骤有助于消除数据中的无关变量（如尺寸、颜色、噪声），并强化模型需要学习的关键特征（如形状、纹理）。

## 2. 核心实现与效果展示

以几张来自 Unsplash 的图片为例。

**原始图像**:
![原始图片](/cv-basics/image-preprocessing/图片1.png) 

### 2.1. 图像读取与尺寸统一

在处理批量图像时，首先需要将它们统一到相同的尺寸，以满足模型输入的要求。使用`cv2.resize()`函数实现。

**可复用代码模块**:
```python
import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('your_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Matplotlib需要RGB格式

# 统一尺寸为 512x512
target_size = (512, 512)
resized_image = cv2.resize(image_rgb, target_size)
```

**效果**:
![图片](/cv-basics/image-preprocessing/图片2.png) 


 ### 2.2. 灰度化 

将彩色图像转换为灰度图像可以大幅减少计算量（从3个颜色通道变为1个），并使得模型更专注于图像的纹理和形状特征，而非颜色信息。

**可复用代码模块**:

```python
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
```

**效果**:
![图片](/cv-basics/image-preprocessing/图片3.png) 
 ### 2.3. 去噪滤波 

图像中的噪声会干扰特征提取。常见的滤波方法：

  - **高斯模糊 (`GaussianBlur`)**: 一种平滑滤波器，能有效抑制高斯噪声。
  - **中值滤波 (`medianBlur`)**: 对去除椒盐噪声（salt-and-pepper noise）特别有效。

**可复用代码模块**:

```python
# 高斯模糊
gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 中值滤波
median_blur = cv2.medianBlur(gray_image, 5)
```

**效果 (高斯模糊)**:

![图片](/cv-basics/image-preprocessing/图片4.png) 
 ### 2.4. 边缘检测 

边缘检测可以提取图像中物体的轮廓信息，这是图像识别和分割任务中的关键特征。这里以经典的**Canny边缘检测算法**为例。

**可复用代码模块**:

```python
# Canny边缘检测
# 两个阈值分别用于强边缘和弱边缘的判断
edges = cv2.Canny(gaussian_blur, threshold1=50, threshold2=150)
```

**效果**:

![图片](/cv-basics/image-preprocessing/图片5.png) 
 ### 2.5. 对比度增强 

为了使图像中的细节更加清晰，可以进行对比度增强。**直方图均衡化** (`equalizeHist`) 是一种常用的全局对比度增强方法，它通过拉伸像素强度的动态范围来改善图像的视觉效果。

**可复用代码模块**:

```python
equalized_image = cv2.equalizeHist(gray_image)
```

**效果**:
![图片](/cv-basics/image-preprocessing/图片6.png) 

 ## 3. 可复用代码文件

  - **完整实现**: [`image_preprocessing.ipynb`](/cv-basics/image-preprocessing/image_preprocessing.ipynb)

