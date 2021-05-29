import os, glob
import torch
import torch.nn as nn
from torch.nn import parameter
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from model import ResNet34


img_list = []
folder = os.getcwd()
print(folder)
img_list.extend(glob.glob(os.path.join(folder,"dataset/mytest","*.jpg")))
# 常见的mode 有 “L” (luminance) 表示灰度图像，“RGB”表示真彩色图像，
#  “CMYK” 表示出版图像，表明图像所使用像素格式。
# im = Image.open(img_list[3]).getchannel(0) # 返回的是L模式下的图片
im = Image.open(img_list[3]).resize((224,224)) # 这里输入一定加（）,为二元组

im = im.convert('RGB') # 强制转换为RGB色彩通道
data_transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
])

imtensor = data_transform(im).reshape(1,3,224,224)
net = ResNet34()
parameters = torch.load("./ResNet34.pth")
net.load_state_dict(parameters)




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