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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class WaveletDataset(Dataset):
	def __init__(self, root_dir, transform=None, rgb=True):
		self.root_dir=root_dir
		self.transform=transform
		
		self.names=np.load(os.path.join(self.root_dir,'processed','names_array.npy'))
		self.data=torch.load(os.path.join(self.root_dir,'processed','image_tensor.pt'))
		
	
	def __len__(self):
		return len(self.names)
	
	def __getitem__(self,idx):

		
		img, name = self.data[idx], self.names[idx]
		img = Image.fromarray(img.numpy())
		
		if self.transform is not None:
			img=self.transorm(img)
			
		if self.target_transform is not None:
            		name = self.target_transform(name)
		
		return img, name
