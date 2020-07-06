from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import glob
from sklearn import preprocessing

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class WaveletDataset(Dataset):
	def __init__(self, root, transform=None, train=True, target_transform=None, temp='/tmp', one_tnsr=False, small=False):
		self.root=root
		self.transform=transform
		self.train=train
		self.target_transform=target_transform
		self.tmp=temp
		self.one_tnsr=one_tnsr
		self.small=small
		self.folder_name='processed_tensors'
		
		self.names=np.load(os.path.join(self.root,'names_array.npy'))
		
		if self.small:
			self.folder_name='small_processed_tensors'
			self.names=np.load(os.path.join(self.root,'small_names_array.npy'))
			
		self.names.sort()
		######
		#Tried giving each one a number, but that doesn't seem to work
		#le = preprocessing.LabelEncoder()
		#self.targets = le.fit_transform(self.names)
		
		#Trying with everything just being ones:
		self.targets=np.ones(len(self.names))
		######
			
		if self.one_tnsr:
			self.data=torch.load(os.path.join(self.root,'image_tensor.pt'))
		else:
			#self.data=os.listdir(os.path.join(self.root,'processed_tensors'))
			self.data=glob.glob(os.path.join(self.root,self.folder_name,'*.pt'))
			self.data.sort()
		
		self.data.sort()
		self.names.sort()
		
	
	def __len__(self):
		return len(self.names)
	
	def __getitem__(self,idx):

		#This one is if you load the whole tensor:
		if self.one_tnsr:
			img, target = self.data[idx], self.targets[idx]
		else:
			img, target = torch.load(os.path.join(self.root, self.folder_name,self.data[idx])), self.targets[idx]
		img = Image.fromarray(img.numpy())
		
		#This is an annoying necessity because PIL doesn't have their 32-float pixel val s*** together
		img=img.convert('L')
		
		if self.transform is not None:
			img=self.transform(img)
			
		if self.target_transform is not None:
            		target = self.target_transform(target)
		
		return img, target
