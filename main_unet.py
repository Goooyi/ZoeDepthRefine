import os
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from HabitatDyndataset.dataset import *
from utils.batch_transform import PILToTensor as bt_PILToTensor
from utils.batch_transform import ToTensor as bt_ToTensor
from utils.custome_loss import MaskedMSELoss, MaskedMSELoss_resized
from utils.dataloader import HabitatDynStreamLoader
from utils.meter import AverageValueMeter

writer = SummaryWriter()
cudnn.benchmark = True  # fast training
# TODO: move np seed to DepthDataset_preload since random sample there
np.random.seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: check train sanity
# TODO: pass parameters like transform using **
def train(model, train_data_dir, valid_loader, num_epochs, run_name, optimizer, scheduler, loss_fun, transform, target_transform, pseudo_depth_transform, mask_transform, device, save_period=10):
    model.to(device)
    min_val_loss = float('inf')

    for epoch in tqdm(range(num_epochs)):
        if epoch % 5 == 0:
            print(f'change the scenes\n')
            train_dataset = DepthDataset_preload(data_dir=train_data_dir, split='', transform=transform, target_transform=target_transform,
                                                 pseudo_depth_transform=pseudo_depth_transform, mask_transform=mask_transform, device=device, sub_video_count=20, cut_off_scene=5*54)
            print(
                f'change the scenes finished data length: {len(train_dataset)}\n')
            train_loader = DataLoader(
                train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=2)

        logs = {}
        # TODO: not do val on every epoch?
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dloader = valid_loader
            # logging
            loss_meter = AverageValueMeter()

            # start = time.time()
            with tqdm(dloader, desc=phase, disable=False) as iterator:
                for input, target in iterator:
                    # print('')
                    # print(phase + ' dataloader time:', time.time() - start)
                    # start = time.time()
                    input, target = input.to(device), target.to(device)
                    # print(phase+' to(cuda) time:', time.time()-start)
                    # start = time.time()
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        prediction = model.forward(input)
                        loss = loss_fun(prediction, target)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        # print(phase+' train 1 batch time: ', time.time()-start)

                        # update loss logs
                        loss_value = loss.cpu().detach().numpy()
                        loss_meter.add(loss_value)
                        loss_logs = {phase + ': ' + loss_fun.__name__: loss_meter.mean}
                        logs.update(loss_logs)

            writer.add_scalar('Loss/' + phase, logs[phase + ': ' + loss_fun.__name__], epoch)
            writer.flush()

            # model save on a iter-base not epoch-base for now
        if  min_val_loss > logs['val: ' + loss_fun.__name__]:
            scheduler.step()
            min_val_loss = logs['val: ' + loss_fun.__name__]
            print(f"min_val_loss: {min_val_loss}")
            torch.save(model, f'./checkpoint/{run_name}/best_model.pth')
            print('Model saved!')


def test():
    pass


if __name__ == '__main__':
    # TODO: load disk to memeory speed to low, multi-?
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
        # transforms.Lambda(lambda x: x.to(device)),
        transforms.Resize(192),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.PILToTensor(),  # using PILToTensor() to load zoe depth unscaled
        # transforms.Lambda(lambda x: x.to(device)),
        transforms.Resize(192),
        # Scale habitatDyn depth pixel values to depth in meters
        transforms.Lambda(lambda t: t/255.*10.)
    ])
    pseudo_depth_transform = transforms.Compose([
        transforms.PILToTensor(),  # using PILToTensor() to load HabitatDyn depth unscaled
        # transforms.Lambda(lambda x: x.to(device)),
        transforms.Resize(192),
        # Scale pseudo depth pixel values to depth in meters
        transforms.Lambda(lambda t: t/256.)
    ])
    mask_transform = transforms.Compose([
        transforms.PILToTensor(),  # using PILToTensor() to load zoe depth unscaled
        # transforms.Lambda(lambda x: x.to(device)),
        transforms.Lambda(lambda x: torch.where(x != 0, 0.8, 0.2)),
        transforms.Resize(192)
    ])

    train_data_dir = '/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_30scenes_newPitch_originalModel'
    test_data_dir = '/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_2scenes_newPitch_3diffModel'

    # train_dataset = DepthDataset_preload(data_dir=train_data_dir, split='', transform=transform, target_transform=target_transform,
                                        #  pseudo_depth_transform=pseudo_depth_transform, mask_transform=mask_transform, device=device, sub_video_count=20,cut_off_scene=5*54) # each scene has 54 videos
    # use a last 20 scenes of total 30 for training
    # frames_for_earch_scene = 13014  # 54 videos

    # train_dataloader = DataLoader(train_dataset, batch_size=64,
    #                               shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=2)

    valid_dataset = DepthDataset_preload(data_dir=test_data_dir, split='', transform=transform, target_transform=target_transform,
                                         pseudo_depth_transform=pseudo_depth_transform, mask_transform=mask_transform, device=device, sub_video_count=10)
    # only use a subset of test_dataset to eval
    # sub_valid_dataset = torch.utils.data.Subset(valid_dataset, range(0, 32)) # type: ignore
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8, drop_last=True)
    # valid_dataloader = DataLoader(sub_valid_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)

    # define loss function
    # loss = MaskedMSELoss_resized()
    list_loss = MaskedMSELoss()
    # create segmentation model with pretrained encoder
    CLASSES = ["all_depth"]
    model = smp.Unet(
        encoder_name='resnet18',
        encoder_weights='imagenet',
        in_channels=4,
        classes=len(CLASSES),
        activation='ReLU',  # could be None for logits or 'softmax2d' for multiclass segmentation
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
    train(model, train_data_dir, valid_dataloader, 1000, run_name, optimizer, lr_scheduler, list_loss, transform=transform,
          target_transform=target_transform, pseudo_depth_transform=pseudo_depth_transform, mask_transform=mask_transform, device=device)

    writer.close()

    # list_train_dataset = ListDepthDataset(data_dir=train_data_dir, split='', transform=list_transform, target_transform=list_target_transform, pseudo_depth_transform=list_pseudo_depth_transform, mask_transform=list_mask_transform, device=device)

    # total_frames = len(list_train_dataset)
    # sub_list_train_dataset = ListSubset(list_train_dataset, range(total_frames-4096, total_frames))

    # list_train_dataloader = HabitatDynStreamLoader(sub_list_train_dataset, bat_size=64, buff_size=16*64, shuffle=True)

    # # transforms for ListDepthDataloader
    # list_transform = transforms.Compose([
    #     bt_ToTensor(),
    #     # transforms.Lambda(lambda x: x.to(device)),
    #     transforms.Resize(192),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],inplace=True)
    # ])
    # list_target_transform = transforms.Compose([
    #     bt_PILToTensor(), # using PILToTensor() to load zoe depth unscaled
    #     # transforms.Lambda(lambda x: x.to(device)),
    #     transforms.Resize(192),
    #     transforms.Lambda(lambda t: t/255.*10.)  # Scale habitatDyn depth pixel values to depth in meters
    # ])
    # list_pseudo_depth_transform = transforms.Compose([
    #     bt_PILToTensor(),# using PILToTensor() to load HabitatDyn depth unscaled
    #     # transforms.Lambda(lambda x: x.to(device)),
    #     transforms.Resize(192),
    #     transforms.Lambda(lambda t: t/256.)  # Scale pseudo depth pixel values to depth in meters
    # ])
    # list_mask_transform = transforms.Compose([
    #     bt_PILToTensor(), # using PILToTensor
    #     # transforms.Lambda(lambda x: x.to(device)),
    #     transforms.Resize(192),
    #     transforms.Lambda(lambda t: t/255.*10.)  # Scale habitatDyn depth pixel values to depth in meters
    # ])

    # # load best saved model checkpoint from the current run
    # if os.path.exists('./checkpoint/best_model.pth'):
    #     best_model = torch.load('./checkpoint/best_model.pth', map_location=DEVICE)
    #     print('Loaded UNet model from this run.')
