import torch
import numpy as np
from .loadHabitatDyn import load_HabitatDyn_img
from HabitatDyndataset.dataset import ListDepthDataset

# TODO: add more functionility like pytorch dataloder
class HabitatDynStreamLoader():
	r"""
	A Stream Data loader for habitatDyn. Differ to the Dataloader class from pytorch, the shuffle is a per-batch shuffle, not the whole dataset like Dataloader.
	
	Args:
		dataset (Dataset): dataset from which to load the data.(img file paths for 
			Habitat Dyn)
		bat_size (int, optional): how many samples per batch to load
		buff_size (int, optinal): how many buffers to pre-load
		shuffle (bool, optional): set to ``True`` to have the data in batch shuffled
		TODO: drop_last configurable (bool, optional): set to ``True`` to drop the last incomplete batch,
			if the dataset size is not divisible by the batch size. If ``False`` and
			the size of dataset is not divisible by the batch size, then the last batch
			will be smaller. (default: ``False``)
		seed (int) set the random state for numpy
	"""
	def __init__(self, dataset, bat_size, buff_size: int, shuffle=False, seed=0):
		# if not isinstance(dataset, ListDepthDataset):
		# 	raise Exception("Wrong Dataset class!")

		if buff_size % bat_size != 0:
			raise Exception("buff must be div by bat_size")
			
		self.dataset = dataset
		self.dataset_len = len(dataset)
		self.dataset_idx = 0 # keep track how many data have been readed
		self.bat_size = bat_size
		self.buff_size = buff_size # a list of numpy array
		self.shuffle = shuffle
		self.seed = seed
		self.rnd = np.random.RandomState(seed)
		self.ptr = 0 # pointer to current x,y in dataset
		
		# init buffer
		self.the_rgb_buffer = [] # list of np rgb image vector
		self.the_pseudo_buffer = [] # list of np pseudo depth image vector
		self.the_depth_buffer = [] # list of numpy depth image vector
		self.the_mask_buffer = [] # list of numpy mask image vector

		self.rgb_mat = None
		self.pseudo_mat = None
		self.depth_mat = None
		self.mask_mat = None

		self.x_data = None
		self.y_data = None
		
		self.reload_buffer() # call to fill the buffer first
		
		
	def reload_buffer(self):
		self.the_rgb_buffer, self.the_pseudo_buffer, self.the_depth_buffer, self.the_mask_buffer = [],[],[],[] # clear buff
		self.ptr = 0
		if self.dataset_idx >= self.dataset_len: # all data readed, epoch finished
			self.dataset_idx = 0
			return -1
		else:
			# drop last
			if self.dataset_idx + self.buff_size > self.dataset_len: # type: ignore
				print("drop last")
				return -2
			else:
				for i in range(self.buff_size):
					rgb_image, pseudo_depth, depth_image, mask = load_HabitatDyn_img(self.dataset, self.dataset_idx + i) # type: ignore
					self.the_rgb_buffer.append(np.array(rgb_image))
					self.the_pseudo_buffer.append(np.array(pseudo_depth))
					self.the_depth_buffer.append(np.array(depth_image))
					self.the_mask_buffer.append(np.array(mask))
				# add up the idx count
				self.dataset_idx += self.buff_size
			
			# TODO impove shuffle
			if self.shuffle:
				self.rnd = np.random.RandomState(self.seed)
				self.rnd.shuffle(self.the_rgb_buffer) # in-place and all with the corresponding order
				self.rnd = np.random.RandomState(self.seed)
				self.rnd.shuffle(self.the_pseudo_buffer) # in-place
				self.rnd = np.random.RandomState(self.seed)
				self.rnd.shuffle(self.the_depth_buffer) # in-place
				self.rnd = np.random.RandomState(self.seed)
				self.rnd.shuffle(self.the_mask_buffer) # in-place
				
			# vectorize
			self.rgb_mat = np.array(self.the_rgb_buffer)
			self.pseudo_mat = np.array(self.the_pseudo_buffer)
			self.depth_mat = np.array(self.the_depth_buffer)
			self.mask_mat = np.array(self.the_mask_buffer)
			
			# resize mask
			self.x_data = torch.cat((self.dataset.transform(self.rgb_mat), self.dataset.pseudo_depth_transform(self.pseudo_mat)), dim=1) # type: ignore
			self.y_data = torch.cat((self.dataset.target_transform(self.mask_mat), self.dataset.mask_transform(self.mask_mat)), dim=1) #type: ignore
			
			return 0
		
	def __iter__(self):
		return self
		
	def __next__(self):
		state = 0 if self.the_rgb_buffer else -2

		if self.ptr + self.bat_size > self.buff_size:
			state = self.reload_buffer() # 0 = success, -1 = hit eof, -2 = not fully loaded
			
		if state == 0:
			start, end = self.ptr, self.ptr + self.bat_size
			
			x, y = self.x_data[start:end,:], self.y_data[start:end,:] #type: ignore
			self.ptr += self.bat_size
			return x,y
			
		# state==-1, all data readed, state == -2, reach EOF, this behave like drop_last = True in pytorch Dataset()
		self.reload_buffer() # prepare for next epoch
		raise StopIteration
		
		
if __name__ == '__main__':
	pass