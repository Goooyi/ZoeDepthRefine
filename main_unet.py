import os

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.models import VGG16_Weights
from torch.utils.tensorboard import SummaryWriter

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
        rgb_image = Image.open(rgb_path).convert('RGB')
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

def train(train_epoch, val_epoch, train_loader, valid_loader, num_epochs, run_name, save_period=10):
    
    writer = SummaryWriter(comment=run_name)
    min_val_loss = float('inf')
    
    # train_logs_list, valid_logs_list = [], []
    for epoch in range(num_epochs):
        print('\nEpoch: {}'.format(epoch))
        train_logs = train_epoch.run(train_loader)
        # print(train_logs)
        valid_logs = valid_epoch.run(valid_loader)
        # train_logs_list.append(train_logs)
        # valid_logs_list.append(valid_logs)
    
        # log tensorboard
        writer.add_scalar('Loss/train', train_logs['maskmse_loss'], epoch)
        writer.add_scalar('Loss/val', valid_logs['maskmse_loss'], epoch)
        
        # Save model if a better loss is obtained
        if min_val_loss > valid_logs['maskmse_loss']:
            min_val_loss = valid_logs['maskmse_loss']
            torch.save(model, f'./checkpoint/{run_name}/best_model.pth')
            print('Model saved!')
            
    writer.close()
            

def test():
    pass


if __name__ == '__main__':
    # TODO: runs too low, check dataset operations
    # TODO: check code do apply mask loss correctly
    # TODO: add argparser: run_name(check_point save folder), train/test
    # DONE: 可以用192*256, 好跑大点的batchsize, check what size does zoe has as default
    # DONE: The problem of using masked label is there are so many frames that does not contain the object so explore other loss or write a custom loss
    # LOG: bs:64,pin_mem=false,num_worker=8 : 1 train epoch(4067 iter):124min, cpu mem: 16G, cpu usage low, gpu mem 7161 gpu usage low
    
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
        transforms.PILToTensor(),
        transforms.Lambda(lambda t: t/255.*10.)  # Scale habitatDyn depth pixel values to depth in meters
    ])
    pseudo_depth_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(192),
        transforms.Lambda(lambda t: t/256.)  # Scale pseudo depth pixel values to depth in meters
    ])
    
    train_data_dir = '/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_30scenes_newPitch_originalModel'
    test_data_dir = '/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_2scenes_newPitch_3diffModel'
    
    train_dataset = DepthDataset(data_dir=train_data_dir, split='', transform=transform, target_transform=target_transform, pseudo_depth_transform=pseudo_depth_transform)
    # use a last 20 scenes of total 30 for training
    total_frames = len(train_dataset)
    frames_for_earch_scene = 13014
    last_n_scenes = 20
    sub_train_dataset = torch.utils.data.Subset(train_dataset, range(total_frames-frames_for_earch_scene*last_n_scenes, total_frames))
    train_dataloader = DataLoader(sub_train_dataset, batch_size=64, shuffle=True, pin_memory=False, num_workers=8)
    
    valid_dataset = DepthDataset(data_dir=test_data_dir, split='', transform=transform, target_transform=target_transform, pseudo_depth_transform=pseudo_depth_transform)
    # only use a subset of test_dataset to eval
    # sub_valid_dataset = torch.utils.data.Subset(valid_dataset, range(0, 256))
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, pin_memory=False, num_workers=8)
    
    ENCODER = 'resnet18'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ["all_depth"]
    # need modify segmentation_models_pytorch to add ReLU activation
    ACTIVATION = 'ReLU' # could be None for logits or 'softmax2d' for multiclass segmentation
    
    # define loss function
    # loss = smp.utils.losses.DiceLoss()
    # loss = smp_utils.losses.MSELoss()
    loss = MaskedMSELoss()
        
    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        in_channels=4,
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )
    
    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])
    
    # define learning rate scheduler (not used )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )    
    
    train_epoch = smp_utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=[],
        optimizer=optimizer,
        device=device,
        verbose=True,
    )
    
    valid_epoch = smp_utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=[],
        device=device,
        verbose=True,
    )

    train(train_epoch, valid_epoch, train_dataloader, valid_dataloader, 200, run_name)
    
    # # load best saved model checkpoint from the current run
    # if os.path.exists('./checkpoint/best_model.pth'):
    #     best_model = torch.load('./checkpoint/best_model.pth', map_location=DEVICE)
    #     print('Loaded UNet model from this run.')
        
