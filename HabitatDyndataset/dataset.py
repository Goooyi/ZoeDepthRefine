import os
import time

import numpy as np
import torch
from PIL import Image
from segmentation_models_pytorch import utils as smp_utils
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import Dataset

# TODO: handle 'device' variable
class ListDepthDataset(Dataset):
		def __init__(self, data_dir, split='', transform=None, target_transform=None, pseudo_depth_transform=None, lambd=0.8):
			self.lambd = lambd
			self.data_dir = data_dir
			self.split = split
			self.transform = transform
			self.target_transform = target_transform
			self.pseudo_depth_transform = pseudo_depth_transform
			self.scene_names = sorted(os.listdir(os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p'))) # 1 video per folder
			self.data = []
		
			for scene_name in self.scene_names:
				rgb_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p', scene_name)
				depth_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/Annotations/480p_depth', scene_name)
				mask_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/Annotations/480p_objectID', scene_name)
				pseudo_depth_folder = os.path.join(data_dir, split, 'zoe_depth_raw', scene_name)
				# prepare one scene time
				self.data.append((rgb_folder, pseudo_depth_folder, depth_folder, mask_folder))
				
		def __len__(self):
			return len(self.data)
		
		def __getitem__(self, index):
			return self.data[index]
				
class DepthDataset(Dataset):
	def __init__(self, data_dir, split='', transform=None, target_transform=None, pseudo_depth_transform=None, lambd=0.8):
		self.lambd = lambd
		self.data_dir = data_dir
		self.split = split
		self.transform = transform
		self.target_transform = target_transform
		self.pseudo_depth_transform = pseudo_depth_transform
		self.scene_names = sorted(os.listdir(os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p'))) # 1 video per folder
		self.data = []

		print(len(self.scene_names))
		for scene_name in self.scene_names[:27]:
			rgb_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p', scene_name)
			depth_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/Annotations/480p_depth', scene_name)
			mask_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/Annotations/480p_objectID', scene_name)
			pseudo_depth_folder = os.path.join(data_dir, split, 'zoe_depth_raw', scene_name)
			# prepare one scene time
			start = time.time()
			for filename in os.listdir(rgb_folder):
				if filename.endswith('.png') or filename.endswith('.jpg'):
					rgb_path = os.path.join(rgb_folder, filename)
					depth_path = os.path.join(depth_folder, filename[:-4]+'.png')
					pseudo_depth_path = os.path.join(pseudo_depth_folder, filename[:-4]+'.png')
					mask_path = os.path.join(mask_folder, filename[:-4]+'.png')
					
					# load to mmemory at first to imporve IO during training
					# load mask
					mask = torch.from_numpy(np.array(Image.open(mask_path).copy())).type(torch.bool).unsqueeze(0).to(device)
					mask = torch.where(mask!=0,self.lambd,1-self.lambd)
					
					# Load RGB image
					rgb_image = np.array(Image.open(rgb_path).convert('RGB').copy())
					depth_image = Image.open(depth_path).copy()
					pseudo_depth = Image.open(pseudo_depth_path).copy()
					if self.transform:
						rgb_image = self.transform(rgb_image)
					if self.target_transform:
						depth_image = self.target_transform(depth_image)
					if self.pseudo_depth_transform:
						pseudo_depth = self.pseudo_depth_transform(pseudo_depth)
					
					# Combine RGB and depth maps into a single input tensor
					# rgb_image = rgb_image.to(device)
					input_tensor = torch.cat((rgb_image, pseudo_depth), dim=0) # type: ignore
					# input_tensor = input_tensor.to(device)
					
					# Load depth maps
					ground_truth_depth = torch.cat((depth_image, mask), dim=0) # type: ignore
					self.data.append((input_tensor, ground_truth_depth))
			print("time to prepare one folder: ", time.time()-start)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		return self.data[index]