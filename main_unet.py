import os
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from HabitatDyndataset.dataset import *
from utils.meter import AverageValueMeter

writer = SummaryWriter()
cudnn.benchmark = True # fast training
np.random.seed(1)

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
            # TODO delete
            if dloader == None:
                continue
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
        transforms.Lambda(lambda x: x.to(device)),
        transforms.Resize(192),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.PILToTensor(), # using PILToTensor() to load zoe depth unscaled
        transforms.Lambda(lambda x: x.to(device)),
        transforms.Lambda(lambda t: t/255.*10.)  # Scale habitatDyn depth pixel values to depth in meters
    ])
    pseudo_depth_transform = transforms.Compose([
        transforms.PILToTensor(),# using PILToTensor() to load HabitatDyn depth unscaled
        transforms.Lambda(lambda x: x.to(device)),
        transforms.Resize(192),
        transforms.Lambda(lambda t: t/256.)  # Scale pseudo depth pixel values to depth in meters
    ])
    
    train_data_dir = '/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_30scenes_newPitch_originalModel'
    test_data_dir = '/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_2scenes_newPitch_3diffModel'
    
    start = time.time()
    train_dataset = DepthDataset(data_dir=train_data_dir, split='', transform=transform, target_transform=target_transform, pseudo_depth_transform=pseudo_depth_transform)
    print("train dataset of x scenes creatation time", time.time()-start)
    # use a last 20 scenes of total 30 for training
    total_frames = len(train_dataset)
    frames_for_earch_scene = 13014 # 53 videos
    frames_for_earch_scene = 10 # 53 videos
    last_n_scenes = 10
    sub_train_dataset = torch.utils.data.Subset(train_dataset, range(total_frames-frames_for_earch_scene*last_n_scenes, total_frames)) # type: ignore
    train_dataloader = DataLoader(sub_train_dataset, batch_size=64, shuffle=True, pin_memory=False, num_workers=4,prefetch_factor=2)
    
    # valid_dataset = DepthDataset(data_dir=test_data_dir, split='', transform=transform, target_transform=target_transform, pseudo_depth_transform=pseudo_depth_transform)
    # # only use a subset of test_dataset to eval
    # sub_valid_dataset = torch.utils.data.Subset(valid_dataset, range(0, 32)) # type: ignore
    # valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)
    # valid_dataloader = DataLoader(sub_valid_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)
    
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
    
    # train(model, train_dataloader, valid_dataloader, 2, run_name, optimizer, loss)
    train(model, train_dataloader, None, 2, run_name, optimizer, loss)
    
    # # load best saved model checkpoint from the current run
    # if os.path.exists('./checkpoint/best_model.pth'):
    #     best_model = torch.load('./checkpoint/best_model.pth', map_location=DEVICE)
    #     print('Loaded UNet model from this run.')
        
    writer = SummaryWriter()
