#-*-coding:utf8-*-

__author__ = "buyizhiyou"
__date__ = "2017-10-18"

import torch
import numpy as np
from torch.autograd import Variable
from data_process import default_loader


classes = ['bus','dinosaur','elephant','flower','horse']
#Predict--------------------------
model = torch.load('alexnet.pkl').cuda()
im = default_loader('473.jpg')
print(im.shape)
im = np.expand_dims(im,0)
#扩展一维，（Ｎ，Ｃ，Ｈ，Ｗ）
im = im.transpose(0,3,1,2)#nhwc===>nchw
im = torch.from_numpy(im).float()
x = Variable(im).cuda()
pred = model(x)
index = torch.max(pred,1)[1].data[0]
print('预测结果:%s'%(classes[index]))