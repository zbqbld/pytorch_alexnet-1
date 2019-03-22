#-*-coding:utf8-*-



from torch.utils.data import Dataset,DataLoader
import numpy as np
from PIL import Image
from  torchvision import transforms,utils

#数据预处理和加载
def default_loader(path):
	im = Image.open(path).convert('RGB')
	im = np.asarray(im.resize((227,227)))
	#把图片转换为二维的array
	return im

#数据集合 将组建dataloder
class MyDataset(Dataset):
	def __init__(self,txt,transform=None,target_transform=None,loader=default_loader):
		f = open(txt,'r')
		self.folder = txt.split('/')[-1].split('.')[0]#也就是train文件夹
		imgs = []
		for line in f.readlines():
			img_name = line.split()[0]
			label = line.split()[1]
			imgs.append((img_name,int(label)))
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self,index):#类的特殊方法
	
		img_name,label = self.imgs[index]
		img_path = './data/'+self.folder+'/'+img_name
		img = self.loader(img_path)#获取图片名再读图片 图片转换为array数组了
		if self.transform is not None:
			img = self.transform(img)
		return img,label

	def __len__(self):
		return len(self.imgs)