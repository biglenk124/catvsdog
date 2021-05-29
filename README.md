## 猫狗识别(ResNet34)

### 前言

- 网络模型：ResNet34、VGG
- 数据集：Kaggle Cat&Dogs
- 实现：自定义数据集、构建模型、学习率衰减、绘制损失函数、评估模型
- 要求：了解pytorch 加载数据集、训练、验证

### 1. 数据处理

> Kaggle官方的猫狗大战数据集。这个数据集包含训练集25000张和测试集12500张。
>

#### 1.1 建立dataset.py文件

**要求**：

- 图片缩放到网络所需要的大小
- 对每张图片建立正确的标签label

#### 1.2 导入需要的包

```python
import os, glob
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
```

重写torch.utils.data.Dataset类下的`__init__`  、` __getitem__` 、` __len__` 三个函数

重写后的数据集类命名为`data_set` ,继承` Dataset`类

#### 1.3 __init__重写

初始化时，需要传入数据所在的位置，图像的变换操作，是否为train。

```python
def __init__(self, folder, transform=None, train=True):
    self.folder = folder
    self.transform = transform
    self.train = train
```

然后，需要空的list存放数据集的图片

`img_list = []`

然后，需要向`img_list`中添加图片信息。需要使用`.extend()`方法，该方法可以在当前list结尾添加一次性添加另一个list中的多个值。

为了将整个文件夹的所有图片信息一次性获取，我们需要使用`glob`模块的`glob`方法，该方法可以返回所有匹配的文件路径列表。

```python
img_list.extend(glob.glob(os.path.join(self.folder,'*.jpg')))
self.img_list = img_list
```

完成` __init__`了， 接下来可以写`__getitem__`了

#### 1.4 __getitem__重写

`__getitem__ ` 函数主要是输入需要获得图片的编号，将`img_list` 对象中的该编号的图片信息导出，这里图片的信息是图片的路径，定义为`img_path`

接着，我们就可以通过`PIL`库中的`Image.open()`函数打开图片了，在这里，我们需要用`.convert("RGB")`来将其强制转换为RGB色彩通道。之后利用之前定义的`transform`对象将读取到的图片转换为网络所需要尺寸的张量：

```python
img = Image.open(img_path).convert("RGB")
img = self.transform(img)
```

接下来，就需要对图片进行打标签操作了。数据集只有训练集是有标签的，因此需要对`train`进行判断，若输入的是训练集，则对图片打上标签后返回图像和标签。

但是，我们在对图像进行测试之后需要生成一个`csv`文件，用于提交到kaggle网站上测试。其中包含了图像的名字和网络预测的标签，因此，在测试集中我们需要返回图像和图像的名字

```python
if self.train:
    if 'dog' in img_path:
        label = 1
    else:
        label = 0
    return img, label
else:
    (_, img_name) = os.path.split(img_path)
    (name, _) = os.path.splitext(img_name)
    return img, name
```

在这里，我们先对`train`对象进行判断:

- 如果为`True`就对图片路径进行判断:若检测到有`dog`的话就打上标签1，没有就打上`0`，然后返回图片和标签；
- 若为`False`就先通过`os.path.split()`将路径与文件名分离，再通过`os.path.splitext()`将图片的名字与后缀分离，最后返回图片和图片的名字。

#### 1.5 __len__重写

现在，我们就写完了`__getitem__`部分。最后，我们需要写`__len__`部分，这边的写法非常简单，只需要使用`len()`函数计算`img_list`的长度后返回即可

#### 1.6 完整代码

```python
import os, glob
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

class data_set(Dataset):
    def __init__(self, folder, transform=None, train=True):
        self.folder = folder
        self.transform = transform
        self.train = train
        img_list = []
        img_list.extend(glob.glob(os.path.join(self.folder,'*.jpg')))
        self.img_list = img_list
    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if self.train:
            if 'dog' in img_path:
                label = 1
            else:
                label = 0
            return img, label
        else:
            (_, img_name) = os.path.split(img_path)
            (name, _) = os.path.splitext(img_name)
            return img, name
    def __len__(self):
        return len(self.img_list)
```

---

### 2. 构建模型

> 本教程使用的模型是ResNet34，该模型由He等在2016年提出，并发表于CVPR2016。
>
> 论文链接：[Deep Residual Learning for Image Recognition](https://link.zhihu.com/?target=https%3A//openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)。
>

#### 2.1 分析模型

网络结构图片：

![image.png](assets/image-20210528102836-sif7qxm.png)

ResNet34一共有33个卷积层和一个全连接层。其中，除了第一个卷积层，其余32个卷积层可以分为16个残差模块，这16个残差模块又可以按照通道数分为4个大模块，即表格中的conv2_x、conv3_x、conv4_x、conv5_x。

#### 2.2 ResBlock

> 关于模型结构化模块化相关的内容可以参考李沐动手学深度学的层和块章节
>

ResNet34模型高度结构化，因此，我们可以把其中的残差模块单独提出来写，这样可以避免大量的重复工作。定义模块和定义网络一样，都包含初始化和前向传播部分。

这里面的残差块分为两种，一种是带下采样的残差块，一种是不带下采样的，分别构建两种残差块（论文作者在这里只构建了一种残差块，通过传入`downsample`来判断是否进行下采样），这两种残差块都需要传入两个参数，即输入通道数和输出通道数。首先定义不带下采样的残差模块：

```python
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
```

用到了一种新的层——批归一化层，该层简称为BN层，在pytorch中，该层的写法和参数如下：

```python
nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
```

> num_features：输入特征图的通道数
>
> eps：稳定系数，默认为1e-5
>
> momentum：动量，用于计算running_mean和running_var的权重，默认为0.1
>
> affine：设置仿射参数是否可学习，设为True表示仿射参数可以通过学习得来，False表示仿射参数为固定值，默认为True
>
> track_running_stats：设置是否追踪均值和方差，设为True表示追踪，False表示不追踪，默认为True
>
> 一般情况下，我们使用`num_features`这个参数。
>
> **官方文档：** [Pytorch-BatchNorm2d](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html%23torch.nn.BatchNorm2d)
>

残差模块内每一层的定义如下：

```python
self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False)
self.bn1 = nn.BatchNorm2d(out_channel)
self.relu1 = nn.ReLU()
self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
self.bn2 = nn.BatchNorm2d(out_channel)
self.relu2 = nn.ReLU()
```

每一层定义完后，就可以写前向传播函数了。在pytorch中，残差连接的前向传播写法是在确保输入和中间某一层输出的张量尺寸完全一致的前提下将二者相加即可。因此，需要一个变量来保存输入的张量，我们将其定义为`res`。

```python
def forward(self, x):
    res = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = x + res     #残差连接
    x = self.relu2(x)
    return x
```

至此，不带下采样的残差模块就写好了。

然后开始写带下采样的残差模块，为了与不带下采样的残差模块区分开，将其命名为`ResBlockDown`。

该模块的第一个卷积层的步长为2，用以缩小特征图的尺寸，同时，在残差连接部分需要使用步长为2的1x1卷积进行下采样，同时提升通道数以和输出的特征图匹配。因此，带下采样的残差模块完整写法如下：

```python
class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlockDown, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 2, 1, bias=False)    #步长为2
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
        self.pool = nn.Conv2d(in_channel, out_channel, 1, 2, 0, bias=False)
    def forward(self, x):
        res = x
        res = self.pool(res)    #对输入进行下采样
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + res
        x = self.relu2(x)
        return x
```

写好了两种残差模块，下面可以实现模型了

#### 2.3 ResNet34

首先，定义模型的名字为`ResNet34`：

需要用到全局平均池化层，又被称作GAP层，在pytorch中，该层的写法和参数如下：

```python
nn.AdaptiveAvgPool2d(output_size)
```

> output_size：输出特征图大小
>
> 一般情况下，我们定义`output_size = 1`，即将特征图缩小至1x1。
>
> **官方文档：** [Pytorch-AdaptiveAvgPool2d](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html%23torch.nn.AdaptiveAvgPool2d)
>

然后，定义其中的第一个卷积和池化层，在这边，输入的图片是3通道的，因此第一个卷积层输入的通道数为3。为了使得输出特征图大小为112x112，我们需要设置边缘补零为3。

池化层的池化核大小为3x3，步长为2，为了使得池化层输出大小为56x56，需要设置池化层边缘补零为1。

另外，卷积层后面有一个BN层和激活层，在这边需要定义，该部分写法如下：

```python
self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
self.bn1 = nn.BatchNorm2d(64)
self.relu1 = nn.ReLU()
self.pool1 = nn.MaxPool2d(3, 2, 1)
```

之后，调用残差模块。为了方便直观，可以调用`nn.Sequential()`

```python
self.conv2 = nn.Sequential(
    ResBlockDown(64, 64),
    ResBlock(64, 64),
    ResBlock(64, 64),
)
self.conv3 = nn.Sequential(
    ResBlockDown(64, 128),
    ResBlock(128, 128),
    ResBlock(128, 128),
    ResBlock(128, 128),
)
self.conv4 = nn.Sequential(
    ResBlockDown(128, 256),
    ResBlock(256, 256),
    ResBlock(256, 256),
    ResBlock(256, 256),
    ResBlock(256, 256),
    ResBlock(256, 256),
)
self.conv5 = nn.Sequential(
    ResBlockDown(256, 512),
    ResBlock(512, 512),
    ResBlock(512, 512),
)
```

ResNet需要进行初始化，在这里可以对卷积核直接使用Xavier初始化方法；

对BN层的权重初始化为1，偏置初始化为0。

#### 2.4 完整代码

```python
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + res     #残差连接
        x = self.relu2(x)
        return x

class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlockDown, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 2, 1, bias=False)    #步长为2
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
        self.pool = nn.Conv2d(in_channel, out_channel, 1, 2, 0, bias=False)
    def forward(self, x):
        res = x
        res = self.pool(res)    #对输入进行下采样
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + res
        x = self.relu2(x)
        return x

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2 = nn.Sequential(
            ResBlockDown(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.conv3 = nn.Sequential(
            ResBlockDown(64, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
        )
        self.conv4 = nn.Sequential(
            ResBlockDown(128, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )
        self.conv5 = nn.Sequential(
            ResBlockDown(256, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
```

**这里，可以把模型打印出来看一下：**

```python
net = ResNet34()
print(net)
```

**结果**

```python
ResNet34(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)   
  (conv2): Sequential(
    (0): ResBlockDown(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
      (pool): Conv2d(64, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
    )
    (1): ResBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (2): ResBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
  )
  (conv3): Sequential(
    (0): ResBlockDown(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
      (pool): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
    )
    (1): ResBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (2): ResBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (3): ResBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
  )
  (conv4): Sequential(
    (0): ResBlockDown(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
      (pool): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
    )
    (1): ResBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (2): ResBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (3): ResBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (4): ResBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (5): ResBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
  )
  (conv5): Sequential(
    (0): ResBlockDown(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
      (pool): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    )
    (1): ResBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (2): ResBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
  )
  (gap): AdaptiveAvgPool2d(output_size=1)
  (fc): Linear(in_features=512, out_features=2, bias=True)
)
```

### 3. 训练模型

#### 3.1 初始化和导入模型

现在，开始训练模型，在文件夹下新建` train.py`文件

```python
# 导包
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

```

然后导入定义的数据集，模型，定义模型为model，并进行初始化

```python
from model import ResNet34
from dataset import data_set
model = ResNet34()
model.weight_init()
```

#### 3.2 定义transform、超参数、数据集

完成以上步骤之后，需要将图片转换为224x224大小的，取值范围在0~1之间的张量。

需要分别使用`transforms.Resize()`和`transforms.ToTensor()`这两个函数实现以上步骤，并通过`transforms.Compose()`将两个步骤整合到一起，并命名为`data_transform`。如下：

```text
data_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])
```

然后，定义我们需要使用的超参数。在这里，我们遍历数据集10轮，批大小为32，初始学习率为0.001，使用交叉熵损失和Adam优化器。

```python
Epoch = 10
batch_size = 32
lr = 0.001
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
```

另外，需要设置学习率衰减，衰减策略为固定步长衰减，学习率每两轮衰减为原来的一半。

在这里，我们需要使用`torch.optim.lr_scheduler.StepLR()`函数，其定义如下：

```text
torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch, verbose)
```

> optimizer：当前使用的优化器
>
> step_size：每过多少次迭代或轮次后进行学习率衰减
>
> gamma：学习率衰减系数，默认为0.1
>
> last_epoch：最后一个轮次的索引，若设为-1表示将初始学习率设置为第一轮的学习率，默认为-1
>
> verbose：如果设为True，则每次更新都会输出一条消息，默认为False
>
> 一般情况下，我们要使用`optimizer, step_size, gamma`这三个参数
>
> **官方文档：** [Pytorch-Optim](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/optim.html)
>

使学习率每两轮衰减为原来的一半，就可以将`step_size`设置为2，`gamma`设置为0.5，即：

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, 2, 0.5)
```

**加载数据集和DataLoader**:

调用之前写的`data_set` 加载训练集和验证集

使用两个线程向模型传送数据：

```python
train_data = data_set("./dataset/train", data_transform, train=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
val_data = data_set("./dataset/validation", data_transform, train=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
# 使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
```

#### 3.3 训练

定义`fit` 函数，判断当前是训练集还是验证集，分别执行相关的操作

这个函数中，我们需要传入模型、DataLoader、以及判断这个数据集是训练集还是验证集的参数

通过`train`参数执行相关的操作，如果该参数为`False`，则需要关闭梯度并冻结BN层的参数更新；如果为`True`则需要启用梯度和BN层的参数更新。

```python
def fit(model, loader, train=True):
    if train:
        torch.set_grad_enabled(True)
        model.train()
    else:
        torch.set_grad_enabled(False)
        model.eval()
```

定义平均损失和准确率，以便观察网络训练情况和画图。定义max_step参数跟踪每一个Epoch中一共迭代了多少次

```python
running_loss = 0.0
acc = 0.0
max_step = 0

```

接下来，我们就要写模型的训练部分了。

在`fit`函数内我们只需要写每一轮的迭代，因此只需要一个for循环即可。

为了使得其输出更加美观，我们可以使用进度条来显示每一轮的迭代进度，并在迭代完成后删除进度条。

在这边通过调用`tqdm`函数实现，并设置`tqdm`的`leave`为`False`：

然后就是训练的步骤了。不过在这里，我们还需要对`train`执行判断，为`True`才执行清空梯度、反向传播和参数更新等操作。并且在该部分需要将计算损失和准确率。

```python
for img, label in tqdm(loader, leave=False):
   max_step += 1
   if train:
       optimizer.zero_grad()

   label_pred = model(img.to(device, torch.float))
   pred = label_pred.argmax(dim=1)
   acc += (pred.data.cpu() == label.data).sum()
   loss = loss_func(label_pred, label.to(device, torch.long))
   running_loss += loss
   if train:
       loss.backward()
       optimizer.step()
```

之后，在一轮结束后计算平均损失和平均准确率。然后再判断一次`train`，若为`True`则调用一次`scheduler.step()`以更新学习率。最后，返回平均损失和平均准确率。

```python
running_loss = running_loss / (max_step)
avg_acc = acc / ((max_step) * batch_size)
if train:
    scheduler.step()
return running_loss, avg_acc
```

完成fit函数后，写train函数。

首先定义训练损失，训练准确率，验证损失，验证准确率四个list，通过一个循环遍历数据集，之后通过fit返回的损失和准确率向四个list中添加。

再输出当前epoch的平均损失和平均准确率。

最后训练完成后使用`model.state_dict()`保存模型参数并返回训练损失，训练准确率，验证损失，验证准确率四个list。

```python
def train():
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    for epoch in range(Epoch):
        train_loss, train_acc = fit(model, train_loader, train=True)
        val_loss, val_acc = fit(model, val_loader, train=False)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_accuracy_list.append(train_acc)
        val_accuracy_list.append(val_acc)
        print('Epoch', epoch + 1, '| train_loss: %.4f' % train_loss, '|train_acc:%.4f' % train_acc, '| validation_loss: %.4f' % val_loss, '|validation_acc:%.4f' % val_acc)
    torch.save(model.state_dict(), "./ResNet34.pth")
    return train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list
```

#### 3.4 绘制损失曲线

在模型训练完成后，我们可以将损失和准确率画成折线图。我们通过定义一个`drew`函数来实现，这个函数传入`train`函数返回的四个list，并使用这4个list来绘制折线图。定义如下：

```python
def drew(train_loss_list, train_acc_list, val_loss_list, val_acc_list):
    plt.figure(figsize = (14,7))
    plt.suptitle("ResNet34 Cats VS Dogs Train & Validation Result")
```

然后通过`plt.subplot()`选择第一个子图用来绘制损失曲线，设置图像的标题、x轴名称、y轴名称。之后就将list中的值传给`plt.plot()`用于绘制图像。最后给图像设置网格并加上图例。如下：

```python
plt.subplot(121)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(Epoch), train_loss_list, label="train")
plt.plot(range(Epoch), val_loss_list, label="validation")
plt.grid(color="k", linestyle=":")
plt.legend()
```

之后选择第二个子图，参照绘制第一个子图的方式绘制准确率曲线。另外，在绘制准确率曲线的时候可以使用`plt.ylim()`来限制y轴的取值范围。

最后，将损失曲线保存成图片并放在result文件夹下，名称为“train result.png”，dpi设置为600。保存完图片后调用`plt.show()`将损失曲线图展示出来。

**主函数**

写一个主函数来调用训练和画图的函数。在这里我们可以直接使用`if __name__ == '__main__':`来实现。如下：

```python
if __name__ == "__main__":
    t_loss, t_acc, v_loss, v_acc = train()
    drew(t_loss, t_acc, v_loss, v_acc)
```

#### 3.5 train.py 完整代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ResNet34
from dataset import data_set

model = ResNet34()
model.weight_init()
data_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])
Epoch = 10
batch_size = 32
lr = 0.001

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, 2, 0.5)
train_data = data_set("./dataset/train", data_transform, train=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
val_data = data_set("./dataset/validation", data_transform, train=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def fit(model, loader, train=True):
    if train:
        torch.set_grad_enabled(True)
        model.train()
    else:
        torch.set_grad_enabled(False)
        model.eval()
    running_loss = 0.0
    acc = 0.0
    max_step = 0
    for img, label in tqdm(loader, leave=False):
        max_step += 1
        if train:
            optimizer.zero_grad()
        label_pred = model(img.to(device, torch.float))
        pred = label_pred.argmax(dim=1)
        acc += (pred.data.cpu() == label.data).sum()
        loss = loss_func(label_pred, label.to(device, torch.long))
        running_loss += loss
        if train:
            loss.backward()
            optimizer.step()
    running_loss = running_loss / (max_step)
    avg_acc = acc / ((max_step) * batch_size)
    if train:
        scheduler.step()
    return running_loss, avg_acc
  
def train():
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    for epoch in range(Epoch):
        train_loss, train_acc = fit(model, train_loader, train=True)
        val_loss, val_acc = fit(model, val_loader, train=False)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_accuracy_list.append(train_acc)
        val_accuracy_list.append(val_acc)
        print('Epoch', epoch + 1, '| train_loss: %.4f' % train_loss, '|train_acc:%.4f' % train_acc, '| validation_loss: %.4f' % val_loss, '|validation_acc:%.4f' % val_acc)
    torch.save(model.state_dict(), "./ResNet34.pth")
    return train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list

def drew(train_loss_list, train_acc_list, val_loss_list, val_acc_list):
    plt.figure(figsize = (14,7))
    plt.suptitle("ResNet34 Cats VS Dogs Train & Validation Result")
    plt.subplot(121)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(Epoch), train_loss_list, label="train")
    plt.plot(range(Epoch), val_loss_list, label="validation")
    plt.grid(color="k", linestyle=":")
    plt.legend()
    plt.subplot(122)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(ymax=1, ymin=0)
    plt.plot(range(Epoch), train_acc_list, label="train")
    plt.plot(range(Epoch), val_acc_list, label="validation")
    plt.grid(color="k", linestyle=":")
    plt.legend()
    plt.savefig("train result.png", dpi=600)
    plt.show()

if __name__ == "__main__":
    t_loss, t_acc, v_loss, v_acc = train()
    drew(t_loss, t_acc, v_loss, v_acc)

```

**如果在运行时报“CUDA out of memory”相关的错误，可通过适当调小batch_size来解决。**

#### 3.6 训练结果

**输出**：

```
Windows PowerShell
版权所有 (C) Microsoft Corporation。保留所有权利。

尝试新的跨平台 PowerShell https://aka.ms/pscore6

PS D:\DeepLearning\projects\CNN_CatvsDog> conda activate pytorch

Epoch 1 | train_loss: 0.6465 |train_acc:0.6408 | validation_loss: 0.5731 |validation_acc:0.7031
Epoch 2 | train_loss: 0.5400 |train_acc:0.7284 | validation_loss: 0.5658 |validation_acc:0.7240
Epoch 3 | train_loss: 0.4370 |train_acc:0.7966 | validation_loss: 0.4310 |validation_acc:0.7881
Epoch 4 | train_loss: 0.3671 |train_acc:0.8353 | validation_loss: 0.3997 |validation_acc:0.8089
Epoch 5 | train_loss: 0.2604 |train_acc:0.8911 | validation_loss: 0.2662 |validation_acc:0.8950
Epoch 6 | train_loss: 0.2112 |train_acc:0.9116 | validation_loss: 0.2073 |validation_acc:0.9131
Epoch 7 | train_loss: 0.1405 |train_acc:0.9435 | validation_loss: 0.1975 |validation_acc:0.9207
Epoch 8 | train_loss: 0.1099 |train_acc:0.9560 | validation_loss: 0.2082 |validation_acc:0.9215
Epoch 9 | train_loss: 0.0581 |train_acc:0.9785 | validation_loss: 0.2559 |validation_acc:0.9223
Epoch 10 | train_loss: 0.0367 |train_acc:0.9871 | validation_loss: 0.2891 |validation_acc:0.9207
PS D:\DeepLearning\projects\CNN_CatvsDog> 
```

**图片**：

![image.png](assets/image-20210528152749-wc05vqm.png)

### 4. 评估模型

#### 4.1 准备工作

新建` evaluate.py` 文件，导入包以及模型

```python
import torch
import numpy
import pandas
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ResNet34
from dataset import data_set
```

选择GPU进行测试

`device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`

由于我们在训练结束后只保存了模型的参数，因此在这里我们需要先导入模型，然后再把参数文件灌入模型内。

导入参数的函数和导入整个模型一样，只不过在导入完参数后要使用`.load_state_dict()`将参数灌入模型内并传送至指定设备。如下所示：

```python
model = ResNet34()
parameters = torch.load('./ResNet34.pth', map_location=torch.device(device))
model.load_state_dict(parameters)
model.to(device)
```

然后，我们需要对图像进行变换，并加载测试集和DataLoader。

```python
data_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])
test_data = data_set("./dataset/test", data_transform, train=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
```

关闭梯度，并冻结BN层的参数。

```python
torch.set_grad_enabled(False)
model.eval()
```

通过`numpy.zeros()`定义一个全零的numpy矩阵用于保存测试结果，该矩阵需要存储图片的名称和模型预测的标签.

因此需要两列，其行数为数据集的大小，在这里我们可以通过`.__len__()`获得。另外，我们需要时设置该矩阵的类型为`int`，如下所示：

```python
result_list = numpy.zeros([test_data.__len__(), 2], dtype=int)
```

#### 4.2 测试

使用进度条来显示测试的进度。同时，也需要定义一个变量来确认当前step数。

```python
step = 0
for img, name in tqdm(test_loader):
```

然后使用模型对图片进行预测并获取预测的标签。

```python
pred = model(img.to(device, torch.float))
label = pred.argmax(dim=1)
```

之后获取图片的名字和预测到的标签转换为`int`型，然后分别放入`result_list`中相应行的第0和第1列。由于`name`是一个元组，因此我们需要取其下标为0的元素。如下所示：

```python
name = int(name[0])
label = int(label)
result_list[step, 0] = name
result_list[step, 1] = label
```

完成以上步骤后对`step`自加一，以进行下一张图片的测试。

```text
step += 1
```

至此，测试部分结束。

#### 4.3 测试结果排序

测试完成后，我们需要先对`result_list`进行一次排序。

由于python读取文件的原因，`result_list`中的第一列是按照类似于`1, 10, 100, 11, 12, 13, ······`的顺序排序的，我们需要将其按照从小到大的顺序排列。

在这里我们利用`.argsort()`对其进行第一列排序，即：`result_list[:, 0].argsort()`。其返回值是一个索引列表，我们直接将这个索引列表放入`result_list`中以进行重构，并更新`result_list`。如下所示：

```python
result_list = result_list[result_list[:, 0].argsort()]
```

#### 4.4 生成csv文件

最后，把测试的结果保存在一个`csv`文件里，将其命名为`submission.csv`并保存在`result`文件夹下。

首先，定义一个字符串数组用于表示我们文件的表头。按照要求，其第一列应为`id`，第二列为`label`，我们就按照这个来定义：

```python
header = ["id","label"]
```

然后，调用`pandas.DataFrame()`将numpy array转换为DataFrame格式，将其`columns`设置为我们定义的表头，`data`设置为`result_list`，如下所示：

```text
csv_data = pandas.DataFrame(columns=header, data=result_list)
```

之后通过`.to_csv()`方法保存成`csv`文件，将其路径设为`./result/submission.csv`，编码格式设置为`utf-8`，同时不启用id索引。即：

```text
csv_data.to_csv("./result/submission.csv", encoding='utf-8', index=False)
```

至此，模型测试完成。

#### 4.5 完整代码

```python
import torch
import numpy
import pandas
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ResNet34
from dataset import data_set

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet34()
parameters = torch.load('./ResNet34.pth', map_location=torch.device(device))
model.load_state_dict(parameters)
model.to(device)
data_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])
test_data = data_set("./dataset/test", data_transform, train=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
torch.set_grad_enabled(False)
model.eval()
result_list = numpy.zeros([test_data.__len__(), 2], dtype=int)
step = 0

for img, name in tqdm(test_loader):
    pred = model(img.to(device, torch.float))
    label = pred.argmax(dim=1)
    name = int(name[0])
    label = int(label)
    result_list[step, 0] = name
    result_list[step, 1] = label
    step += 1

result_list = result_list[result_list[:, 0].argsort()]
header = ["id", "label"]
csv_data = pandas.DataFrame(columns=header, data=result_list)
csv_data.to_csv("./result/submission.csv", encoding='utf-8', index=False)
```

```

```

### 总结

#### K折交叉验证

> **本次模型不足**
>
> 将训练集的一部分图片划分成为验证集，在这里按照10：1的比例划分的验证集。
>

一般情况将K折交叉验证用于模型调优，找到使得模型泛化性能最优的超参值。找到后，在全部训练集上重新训练模型，并使用独立测试集对模型性能做出最终评价。

K折交叉验证使用了无重复抽样技术的好处：每次迭代过程中每个样本点只有一次被划入训练集或测试集的机会。

## 闲散地

> 自由发挥，测试脚本里的一些函数
>

### dataset.py

#### init

```python
# glob
img_list = []
folder = os.getcwd()
print(folder)
img_list.extend(glob.glob(os.path.join(folder,"*.pth")))
print(img_list)

# 输出内容
# D:\DeepLearning\projects\CNN_CatvsDog
# ['D:\\DeepLearning\\projects\\CNN_CatvsDog\\ResNet34.pth']
```

```python
# 如果文件地址不存在，虽然也不会报错，但是img_list就是空的
imglist = data_set("./dataset/mytest") 
print(imglist)

#输出
#['./dataset/mytest\\cat.0.jpg', './dataset/mytest\\cat.1251.jpg', 
# './dataset/mytest\\dog.3900.jpg', './dataset/mytest\\dog.3901.jpg']
# <__main__.data_set object at 0x000002947CF14BE0>
```

#### getitem

```python

# 常见的mode 有 “L” (luminance) 表示灰度图像，“RGB”表示真彩色图像，
#  “CMYK” 表示出版图像，表明图像所使用像素格式。
# im = Image.open(img_list[3]).getchannel(0) # 返回的是L模式下的图片
im = Image.open(img_list[1]).resize((100,100)) # 这里输入一定加（）,为二元组
# im.show()
print(im)
print(im.getbbox()) # (0, 0, 100, 100) 返回四元组，左上右下坐标
r,g,b = im.split()
r.show(),g.show(),b.show()
```

```python
im = im.convert('RGB') # 强制转换为RGB色彩通道
data_transform = transforms.Compose([
    transforms.Resize([3, 3]),
    transforms.ToTensor(),
])
imtensor = data_transform(im)

print(imtensor) # 会打印一个3x3x3的tensor
print(imtensor.shape) # torch.size([3,3,3])
```

```python
path = "C:/ProgramFiles/Anaconda/envs/pytorch/python.exe"
(_,name) = os.path.split(path)
print(name)
# python.exe
(_,name) = os.path.splitext(path)
print(name)
# .exe
```

### 各层输出

#### conv1输出

```python
im = im.convert('RGB') # 强制转换为RGB色彩通道
data_transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
])

imtensor = data_transform(im).reshape(1,3,224,224)
net = ResNet34()
parameters = torch.load("./ResNet34.pth")
net.load_state_dict(parameters)


a = net.conv1(imtensor) # a.shpae :torch.Size([1, 64, 112, 112])
a = a.reshape(64,112,112)
unloader = transforms.ToPILImage()


for i in range(len(a)): # 因为不加range,i会变成iter
    im = unloader(a[i])
    im.save(f"./dataset/mytest/dog3901/conv1_{i}.jpg")


im = unloader(a[1])
im.show()
```

![image.png](assets/image-20210529165519-nwzbo5w.png)

**relu1**

torch.Size([1, 64, 112, 112])

![image.png](assets/image-20210529165532-n1bmx6d.png)

**pool1**

torch.Size([1, 64, 56, 56])

![image.png](assets/image-20210529165554-4tz455f.png)

### conv2输出

torch.Size([1, 64, 28, 28])

![image.png](assets/image-20210529165607-0limn5j.png)

### conv3输出

torch.Size([1, 128, 14, 14])

```python

model = nn.Sequential(net.conv1,net.bn1,net.relu1,net.pool1,net.conv2,net.conv3)
a = model(imtensor)
print(a.shape)
a = a.reshape(128,14,14)

# 保存图片到本地
def savepic():
    unloader = transforms.ToPILImage()
    for i in range(len(a)): # 因为不加range,i会变成iter
        im = unloader(a[i])
        im.save(f"./dataset/mytest/dog3901/rh.conv3_{i}.jpg")

    im = unloader(a[1])
    im.show()

savepic()
```

![image.png](assets/image-20210529165647-nm7nb8e.png)
