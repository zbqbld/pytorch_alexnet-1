#-*-coding:utf8-*-

__author__ = 'buyizhiyou'
__date__ = '2017-10-18'


import torch
import torch.nn  as nn
from  torchvision import transforms,utils
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import pdb


#建立模型
class AlexNet(nn.Module):

	def __init__(self,num_classes=5):
		super(AlexNet,self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
			nn.ReLU(inplace=True),#inplace:原地　　不创建新对象，直接对传入数据进行修改
			nn.MaxPool2d(kernel_size=3,stride=2),
			nn.Conv2d(64,192,kernel_size=5,padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3,stride=2),
			nn.Conv2d(192,384,kernel_size=3,padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384,256,kernel_size=3,padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256,256,kernel_size=3,padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3,stride=2),
			)
		self.classfier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256*6*6,4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096,4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096,num_classes),
			)

	def forward(self,x):
		x = self.features(x)
		x = x.view(x.size(0),256*6*6)#view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
		#print(x) 没有用softmax?
		x = self.classfier(x)
		return x

#另一种方式建立模型
# class AlexNet(nn.Module):
# 	def __init__(self,num_classes=5):
# 		super(AlexNet,self).__init__()

# 		self.features = nn.Sequential()
# 		self.features.add_module('conv1',nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2))
# 		self.features.add_module('relu1',nn.ReLU(inplace=True))
# 		self.features.add_module('pool1',nn.MaxPool2d(kernel_size=3,stride=2))
# 		self.features.add_module('conv2',nn.Conv2d(64,192,kernel_size=5,padding=2))
# 		self.features.add_module('relu2',nn.ReLU(inplace=True))
# 		self.features.add_module('pool2',nn.MaxPool2d(kernel_size=3,stride=2))
# 		self.features.add_module('conv3',nn.Conv2d(192,384,kernel_size=3,padding=1))
# 		self.features.add_module('relu3',nn.ReLU(inplace=True))
# 		self.features.add_module('conv4',nn.Conv2d(384,256,kernel_size=3,padding=1))
# 		self.features.add_module('relu4',nn.ReLU(inplace=True))
# 		self.features.add_module('conv5',nn.Conv2d(256,256,kernel_size=3,padding=1))
# 		self.features.add_module('pool5',nn.MaxPool2d(kernel_size=3,stride=2))

# 		self.classfier = nn.Sequential()
# 		self.classfier.add_module('fc6',nn.Linear(256*6*6,4096))
# 		self.classfier.add_module('relu6',nn.ReLU(inplace=True))
# 		self.classfier.add_module('dropout6',nn.Dropout())	
# 		self.classfier.add_module('fc7',nn.Linear(4096,4096))
# 		self.classfier.add_module('relu7',nn.ReLU(inplace=True))
# 		self.classfier.add_module('dropout7',nn.Dropout())
# 		self.classfier.add_module('fc8',nn.Linear(4096,num_classes))

# 	def forward(self,x):
# 		x = self.features(x)
# 		x = x.view(x.size(0),256*6*6)
# 		x = self.classfier(x)
# 		return x
'''
AlexNet (
  (features): Sequential (
    (conv1): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (relu1): ReLU (inplace)
    (pool1): MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1))
    (conv2): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (relu2): ReLU (inplace)
    (pool2): MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1))
    (conv3): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu3): ReLU (inplace)
    (conv4): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu4): ReLU (inplace)
    (conv5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (pool5): MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1))
  )
  (classfier): Sequential (
    (fc6): Linear (9216 -> 4096)
    (relu6): ReLU (inplace)
    (dropout6): Dropout (p = 0.5)
    (fc7): Linear (4096 -> 4096)
    (relu7): ReLU (inplace)
    (dropout7): Dropout (p = 0.5)
    (fc8): Linear (4096 -> 5)
  )
)
'''



