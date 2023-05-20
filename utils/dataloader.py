import torch
import numpy as np
from loadHabitatDyn import load_HabitatDyn

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
	def __init__(self, dataset, bat_size, buff_size, shuffle=False, seed=0):
		if buff_size % bat_size != 0:
			raise Exception("buff must be div by bat_size")
			
		self.dataset = dataset
		self.dataset_len = len(dataset)
		self.dataset_idx = 0 # keep track how many data have been readed
		self.bat_size = bat_size
		self.buff_size = buff_size # a list of numpy array
		self.shuffle = shuffle
		self.rnd = np.random.RandomState(seed)
		self.ptr = 0 # pointer to current x,y in dataset
		
		# init buffer
		self.the_buffer = [] # list of numpy image vector
		self.xy_mat = None
		self.x_data = None
		self.y_data = None
		
		self.reload_buffer() # call to fill the buffer first
		
	# TODO：input of init?
	def input_transform():
		pass
	def target_transform():
		pass
	def mask_transform():
		pass
		
	def reload_buffer(self):
		self.the_buffer = []
		self.ptr = 0
		if self.dataset_idx >= self.dataset_len: # all data readed, epoch finished
			self.dataset_idx = 0
			return -1
		else:
			# TODO implement load_HabitatDyn: return input,target,mask, take care the size so self.xy_mat will vectorize better
			xy_data = load_HabitatDyn(self.dataset, self.dataset_idx, self.buff_size)
			if xy_data:
				self.buff_size.append(xy_data)
			else:
				return -2
			
			if self.shuffle:
				self.rnd.shuffle(self.the_buffer) # in-place
				
			# TODO: is this step necessary?
			self.xy_mat = np.array(self.the_buffer)
			
			# TODO: trans_froms, move to gpu to do preprocessing?
			self.x_data = ...
			self.y_data = ...
			
			return 0
		
	def __iter__(self):
		return self
		
	def __next__(self):
		state = 0
		if self.ptr + self.bat_size > self.buff_size:
			print(" ** reloading buffer ** ")
			state = self.reload_buffer() # 0 = success, -1 = hit eof, -2 = not fully loaded
			
		if state == 0:
			start, end = self.ptr, self.ptr + self.bat_size
			
			# TODO load utils to load x and y
			x, y = self.x_data[start:end,:], self.y_data[start:end,:]
			self.ptr += self.bat_size
			return x,y
			
		# state == -2, reach EOF, this behave like drop_last = True in pytorch Dataset()
		self.reload_buffer() # prepare for next epoch
		raise StopIteration
		
		
		
		
		
		
		
		
		
		
		
		
if __name__ == '__main__':
	pass