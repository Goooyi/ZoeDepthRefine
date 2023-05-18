import os
import time
from utils.meter import AverageValueMeter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from segmentation_models_pytorch import utils as smp_utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

writer = SummaryWriter()
cudnn.benchmark = True # fast training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.__name__ = "maskmse_loss"
    
    def forward(self, input, target):
        # Calculate the masked MSE loss
        input_intp = F.interpolate(input, size=(480, 640), mode="bilinear")
        mse_loss = (input_intp[:,0,:,:] - target[:,0,:,:]) ** 2
        mse_loss *= target[:,1,:,:]
        mse_loss = torch.mean(mse_loss)
    
        return mse_loss
    
class DepthDataset(Dataset):
    def __init__(self, data_dir, split='', transform=None, target_transform=None, pseudo_depth_transform=None, lambd=0.8):
        self.lambd = lambd
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.pseudo_depth_transform = pseudo_depth_transform
        self.scene_names = sorted(os.listdir(os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p')))
        self.data = []

        for scene_name in self.scene_names:
            rgb_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p', scene_name)
            depth_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/Annotations/480p_depth', scene_name)
            mask_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/Annotations/480p_objectID', scene_name)
            pseudo_depth_folder = os.path.join(data_dir, split, 'zoe_depth_raw', scene_name)
            for filename in os.listdir(rgb_folder):
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    rgb_path = os.path.join(rgb_folder, filename)
                    depth_path = os.path.join(depth_folder, filename[:-4]+'.png')
                    pseudo_depth_path = os.path.join(pseudo_depth_folder, filename[:-4]+'.png')
                    mask_path = os.path.join(mask_folder, filename[:-4]+'.png')
                    self.data.append((rgb_path, depth_path, pseudo_depth_path, mask_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rgb_path, depth_path, pseudo_depth_path, mask_path = self.data[index]
        # TODO write more compact transform: like use from torchvision.io import read_image, but with caution 0 255
        
        # load mask
        mask = torch.from_numpy(np.array(Image.open(mask_path))).type(torch.bool).unsqueeze(0)
        # mask.to(device)
        mask = torch.where(mask!=0,self.lambd,1-self.lambd)

        # Load RGB image
        rgb_image = np.array(Image.open(rgb_path).convert('RGB'))
        
        depth_image = Image.open(depth_path)
        pseudo_depth = Image.open(pseudo_depth_path)
        if self.transform:
            rgb_image = self.transform(rgb_image)
        if self.target_transform:
            ground_truth_depth = self.target_transform(depth_image)
        if self.pseudo_depth_transform:
            pseudo_depth = self.pseudo_depth_transform(pseudo_depth)

        # Combine RGB and depth maps into a single input tensor
        # rgb_image = rgb_image.to(device)
        input_tensor = torch.cat((rgb_image, pseudo_depth), dim=0) # type: ignore
        # input_tensor = input_tensor.to(device)
        
        # Load depth maps
        ground_truth_depth = torch.cat((ground_truth_depth, mask), dim=0)
        return input_tensor, ground_truth_depth

# TODO: check train sanity
def train(model, train_loader, valid_loader, num_epochs, run_name, optimizer, loss_fun, save_period=10):
    model.to(device)
    min_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dloader = valid_loader
            # logging
            logs = {}
            loss_meter = AverageValueMeter()
            
            start = time.time()
            with tqdm(dloader, desc = phase) as iterator:
                for input, target in iterator:
                    print(phase + ' dataloader time:', time.time() - start)
                    start = time.time()
                    input, target = input.to(device), target.to(device)
                    print(phase+' to(cuda) time:', time.time()-start)
                    start = time.time()
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        prediction = model.forward(input)
                        loss = loss_fun(prediction, target) 
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        print(phase+' train 1 batch time: ', time.time()-start)

                        # update loss logs
                        loss_value = loss.cpu().detach().numpy()
                        loss_meter.add(loss_value)
                        loss_logs = {loss_fun.__name__: loss_meter.mean}
                        logs.update(loss_logs)

            loss_name = 'Loss/' + phase
            writer.add_scalar(loss_name, logs['maskmse_loss'], epoch)
            writer.flush()

            if phase == 'train' and min_val_loss > logs['maskmse_loss']:
                min_val_loss = logs['maskmse_loss']
                torch.save(model, f'./checkpoint/{run_name}/best_model.pth')
                print('Model saved!')
            

def test():
    pass


if __name__ == '__main__':
    # TODO: runs too low, check dataset operations
    # TODO: check code do apply mask loss correctly
    # TODO: add argparser: run_name(check_point save folder), train/test
    # DONE: 可以用192*256, 好跑大点的batchsize, check what size does zoe has as default
    # DONE: The problem of using masked label is there are so many frames that does not contain the object so explore other loss or write a custom loss
    # LOG: bs:64,pin_mem=false,num_worker=8 : 1 train epoch(4067 iter):124min, cpu mem: 4G, cpu usage low, gpu mem 7161 gpu usage low
    
    run_name = '30scene_ori_Epoch200_MaskedMSEloss_lambda8'
    path = f'./checkpoint/{run_name}'
    if not os.path.exists(path):
        os.makedirs(path)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(192),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.PILToTensor(), # using PILToTensor() to load zoe depth unscaled
        transforms.Lambda(lambda t: t/255.*10.)  # Scale habitatDyn depth pixel values to depth in meters
    ])
    pseudo_depth_transform = transforms.Compose([
        transforms.PILToTensor(),# using PILToTensor() to load HabitatDyn depth unscaled
        transforms.Resize(192),
        transforms.Lambda(lambda t: t/256.)  # Scale pseudo depth pixel values to depth in meters
    ])
    
    train_data_dir = '/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_30scenes_newPitch_originalModel'
    test_data_dir = '/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_2scenes_newPitch_3diffModel'
    
    train_dataset = DepthDataset(data_dir=train_data_dir, split='', transform=transform, target_transform=target_transform, pseudo_depth_transform=pseudo_depth_transform)
    # use a last 20 scenes of total 30 for training
    total_frames = len(train_dataset)
    frames_for_earch_scene = 13014
    frames_for_earch_scene = 10
    last_n_scenes = 20
    sub_train_dataset = torch.utils.data.Subset(train_dataset, range(total_frames-frames_for_earch_scene*last_n_scenes, total_frames))
    train_dataloader = DataLoader(sub_train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4,prefetch_factor=2)
    
    valid_dataset = DepthDataset(data_dir=test_data_dir, split='', transform=transform, target_transform=target_transform, pseudo_depth_transform=pseudo_depth_transform)
    # only use a subset of test_dataset to eval
    sub_valid_dataset = torch.utils.data.Subset(valid_dataset, range(0, 32))
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(sub_valid_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)
    
    # define loss function
    loss = MaskedMSELoss()
    # create segmentation model with pretrained encoder
    CLASSES = ["all_depth"]
    model = smp.Unet(
        encoder_name='resnet18', 
        encoder_weights='imagenet', 
        in_channels=4,
        classes=len(CLASSES), 
        activation='ReLU', # could be None for logits or 'softmax2d' for multiclass segmentation
    )
    
    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])
    
    # define learning rate scheduler (not used )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )    
    
    train(model, train_dataloader, valid_dataloader, 2, run_name, optimizer, loss)
    
    # # load best saved model checkpoint from the current run
    # if os.path.exists('./checkpoint/best_model.pth'):
    #     best_model = torch.load('./checkpoint/best_model.pth', map_location=DEVICE)
    #     print('Loaded UNet model from this run.')
        
    writer = SummaryWriter()
