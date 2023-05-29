import os
import random

import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset


# TODO: handle 'device' variable
# TODO: delete 'lambd'
class ListDepthDataset(Dataset):
	"""Not loading data to memory at all in this class, only generate a list of data path
	   used for stream dataloader
	"""
	def __init__(self, data_dir, device, split='', transform=None, target_transform=None, pseudo_depth_transform=None, mask_transform=None, lambd=0.8):
		self.device = device
		self.lambd = lambd
		self.data_dir = data_dir
		self.split = split
		self.transform = transform
		self.target_transform = target_transform
		self.pseudo_depth_transform = pseudo_depth_transform
		self.mask_transform = mask_transform
		self.scene_names = sorted(os.listdir(os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p'))) # 1 video per folder
		self.data = []
	
		for scene_name in self.scene_names:
			rgb_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p', scene_name)
			depth_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/Annotations/480p_depth', scene_name)
			mask_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/Annotations/480p_objectID', scene_name)
			pseudo_depth_folder = os.path.join(data_dir, split, 'zoe_depth_raw', scene_name)
			# prepare one scene time
			for filename in os.listdir(rgb_folder):
				if filename.endswith('.png') or filename.endswith('.jpg'):
					rgb_path = os.path.join(rgb_folder, filename)
					depth_path = os.path.join(depth_folder, filename[:-4]+'.png')
					pseudo_depth_path = os.path.join(pseudo_depth_folder, filename[:-4]+'.png')
					mask_path = os.path.join(mask_folder, filename[:-4]+'.png')
					self.data.append((rgb_path, pseudo_depth_path, depth_path, mask_path))
			
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):
		return self.data[index]
				

class DepthDataset(Dataset):
	def __init__(self, data_dir, device, split='', transform=None, target_transform=None, pseudo_depth_transform=None, mask_transform=None, lambd=0.8):
		self.device = device
		self.lambd = lambd
		self.data_dir = data_dir
		self.split = split
		self.transform = transform
		self.target_transform = target_transform
		self.pseudo_depth_transform = pseudo_depth_transform
		self.mask_transform = mask_transform
		self.scene_names = sorted(os.listdir(os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p'))) # 1 video per folder
		self.data = []

		for iidx, scene_name in enumerate(self.scene_names):
			rgb_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p', scene_name)
			depth_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/Annotations/480p_depth', scene_name)
			mask_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/Annotations/480p_objectID', scene_name)
			pseudo_depth_folder = os.path.join(data_dir, split, 'zoe_depth_raw', scene_name)
			# prepare one scene time
			for filename in os.listdir(rgb_folder):
				if filename.endswith('.png') or filename.endswith('.jpg'):
					rgb_path = os.path.join(rgb_folder, filename)
					depth_path = os.path.join(depth_folder, filename[:-4]+'.png')
					pseudo_depth_path = os.path.join(pseudo_depth_folder, filename[:-4]+'.png')
					mask_path = os.path.join(mask_folder, filename[:-4]+'.png')
					self.data.append((rgb_path,pseudo_depth_path,depth_path,mask_path))
					
	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		# Load RGB image
		rgb_path,pseudo_depth_path,depth_path,mask_path = self.data[index]
		rgb_image = Image.open(rgb_path).convert('RGB').copy()
		depth_image = Image.open(depth_path).copy()
		pseudo_depth = Image.open(pseudo_depth_path).copy()
		mask = Image.open(mask_path).copy()

		if self.transform:
			rgb_image = self.transform(rgb_image) # shape 3*192*256
		if self.target_transform:
			depth_image = self.target_transform(depth_image) # shape 1*192*256
		if self.pseudo_depth_transform:
			pseudo_depth = self.pseudo_depth_transform(pseudo_depth) # shape 1*192*256
		if self.mask_transform:
			mask = self.mask_transform(mask) # shape 
		
		# Combine RGB and depth maps into a single input tensor
		input_tensor = torch.cat((rgb_image, pseudo_depth), dim=0) # type: ignore
		# Load depth maps
		ground_truth_depth = torch.cat((depth_image, mask), dim=0) # type: ignore

		self.data.append((input_tensor, ground_truth_depth))
		return self.data[index]


# TODO: device not used for now
class DepthDataset_preload(Dataset):
	def __init__(self, data_dir, device, split='', transform=None, target_transform=None, pseudo_depth_transform=None, mask_transform=None, lambd=0.8, sub_video_count=None, cut_off_scene=None):
		self.device = device
		self.lambd = lambd
		self.data_dir = data_dir
		self.split = split
		self.transform = transform
		self.target_transform = target_transform
		self.pseudo_depth_transform = pseudo_depth_transform
		self.mask_transform = mask_transform
		self.scene_names = sorted(os.listdir(os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p'))) # 1 video per folder
		self.data = []

		if cut_off_scene:
			self.scene_names = self.scene_names[cut_off_scene:]
		if sub_video_count:
			self.scene_names = random.sample(self.scene_names, sub_video_count)
		for scene_name in tqdm(self.scene_names, desc='load dataset'):
			rgb_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p', scene_name)
			depth_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/Annotations/480p_depth', scene_name)
			mask_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/Annotations/480p_objectID', scene_name)
			pseudo_depth_folder = os.path.join(data_dir, split, 'zoe_depth_raw', scene_name)
			# prepare one scene time

			for filename in os.listdir(rgb_folder):
				if filename.endswith('.png') or filename.endswith('.jpg'):
					rgb_path = os.path.join(rgb_folder, filename)
					depth_path = os.path.join(depth_folder, filename[:-4]+'.png')
					pseudo_depth_path = os.path.join(pseudo_depth_folder, filename[:-4]+'.png')
					mask_path = os.path.join(mask_folder, filename[:-4]+'.png')
					
					# Load RGB image
					rgb_image = Image.open(rgb_path).convert('RGB').copy() # shape 480*640*3
					depth_image = Image.open(depth_path).copy() # shape 480*640
					pseudo_depth = Image.open(pseudo_depth_path).copy() # shape 480*640
					mask = Image.open(mask_path).copy()

					if self.transform:
						rgb_image = self.transform(rgb_image) # shape 3*192*256
					if self.target_transform:
						depth_image = self.target_transform(depth_image) # shape 1*192*256
					if self.pseudo_depth_transform:
						pseudo_depth = self.pseudo_depth_transform(pseudo_depth) # shape 1*192*256
					if self.mask_transform:
						mask = self.mask_transform(mask) # shape 
					
					# Combine RGB and depth maps into a single input tensor
					input_tensor = torch.cat((rgb_image, pseudo_depth), dim=0) # type: ignore
					# Load depth maps
					ground_truth_depth = torch.cat((depth_image, mask), dim=0) # type: ignore

					self.data.append((input_tensor, ground_truth_depth))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		return self.data[index]
	

class ListSubset(ListDepthDataset):
	r"""
	Subset of a dataset at specified indices.

	Args:
		dataset (Dataset): The whole Dataset
		indices (sequence): Indices in the whole set selected for subset
	"""

	def __init__(self, dataset, indices):
		self.dataset = dataset
		self.indices = indices

		self.device = dataset.device
		self.lambd = dataset.lambd
		self.transform = dataset.transform
		self.target_transform = dataset.target_transform
		self.pseudo_depth_transform = dataset.pseudo_depth_transform
		self.mask_transform = dataset.mask_transform
		self.data = dataset.data[indices[0]:indices[-1]]

	def __getitem__(self, idx):
		return self.dataset[idx]

	def __len__(self):
		return len(self.indices)