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
from torch.utils.tensorboard import SummaryWriter #type: ignore
from tqdm import tqdm

from HabitatDyndataset.dataset import *
from utils.batch_transform import PILToTensor as bt_PILToTensor
from utils.batch_transform import ToTensor as bt_ToTensor
from utils.custome_loss import MaskedMSELoss, MaskedMSELoss_resized
from utils.dataloader import HabitatDynStreamLoader
from utils.meter import AverageValueMeter

import ray
from ray import air, tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

from ray.tune.search.hyperopt import HyperOptSearch

cudnn.benchmark = True  # fast training
# TODO: move np seed to DepthDataset_preload since random sample there
np.random.seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ray.init(num_gpus=1)


# TODO: check train sanity
# TODO: pass parameters like transform using **
# TODO: transfer stream dataloader structure to dataset
def train(config, model, train_data_dir, valid_dataloader, num_epochs, run_name, transform, target_transform, pseudo_depth_transform, mask_transform):
# def train(params={}):
    min_metric_loss = float('inf')
    metric = MaskedMSELoss(1.0, "movingObject_MSE")
    mname = metric.name if metric.name else ''
    loss_fun = MaskedMSELoss(config['lambd'])
    optimizer = torch.optim.Adam([
       dict(params=model.parameters(), lr=config["lr"]),
    ])

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    for epoch in tqdm(range(start_epoch, num_epochs)):
        if epoch % 5 == 0:
            print(f'change the scenes\n')
            train_dataset = DepthDataset_preload(data_dir=train_data_dir, split='', transform=transform, target_transform=target_transform,
                                                 pseudo_depth_transform=pseudo_depth_transform, mask_transform=mask_transform, device=device, sub_video_count=20, cut_off_scene=5*54)
            print(
                f'change the scenes finished data length: {len(train_dataset)}\n')
            train_loader = DataLoader(
                train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=2)

        logs = {}
        metrics = {}
        # TODO: not do val on every epoch?
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dloader = train_loader # type: ignore
            else:
                model.eval()   # Set model to evaluate mode
                dloader = valid_dataloader
            # logging
            loss_meter = AverageValueMeter()
            metric_meter = AverageValueMeter()

            # start = time.time()
            with tqdm(dloader, desc=phase, disable=False) as iterator:
                for input, target in iterator:
                    input, target = input.to(device), target.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        prediction = model.forward(input)
                        loss = loss_fun(prediction, target)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        metric_value = metric(prediction, target)

                        # update loss logs
                        loss_value = loss.cpu().detach().numpy()
                        loss_meter.add(loss_value)
                        loss_logs = {phase + ': ' + loss_fun.__name__: loss_meter.mean}
                        logs.update(loss_logs)
                        # update metric logs
                        metric_value = metric_value.cpu().detach().numpy()
                        metric_meter.add(metric_value)
                        metric_logs = {phase+': ' + mname: metric_meter.mean}
                        metrics.update(metric_logs)

            # writer.add_scalar('Loss/' + phase, logs[phase + ': ' + loss_fun.__name__], epoch)
            # writer.flush()

            # model save on a iter-base not epoch-base for now
        if  min_metric_loss > metrics['val: ' + mname]:
            min_metric_loss = metrics['val: ' + mname]
            print(f"min_metric_loss: {min_metric_loss}")
            torch.save(model, f'./checkpoint/{run_name}/best_model.pth')
            print('Model saved!')

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"object_val_loss": metrics['val: ' + mname]},
            checkpoint=checkpoint,
        )



def test():
    pass


if __name__ == '__main__':
    # TODO: load disk to memeory speed to low, multi-?
    # TODO: check code do apply mask loss correctly
    # TODO: add argparser: run_name(check_point save folder), train/test
    # DONE: 可以用192*256, 好跑大点的batchsize, check what size does zoe has as default
    # DONE: The problem of using masked label is there are so many frames that does not contain the object so explore other loss or write a custom loss
    # LOG: bs:64,pin_mem=false,num_worker=8 : 1 train epoch(4067 iter):124min, cpu mem: 4G, cpu usage low, gpu mem 7161 gpu usage low

    run_name = '30scene_ori_Epoch1000_MaskedMSEloss_lambda6_worker8_cosAneal12'

    

    path = f'./checkpoint/{run_name}'
    if not os.path.exists(path):
        os.makedirs(path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(192),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.PILToTensor(),  # using PILToTensor() to load zoe depth unscaled
        transforms.Resize(192),
        transforms.Lambda(lambda t: t/255.*10.)
    ])
    pseudo_depth_transform = transforms.Compose([
        transforms.PILToTensor(),  # using PILToTensor() to load HabitatDyn depth unscaled
        transforms.Resize(192),
        transforms.Lambda(lambda t: t/256.)
    ])
    mask_transform = transforms.Compose([
        transforms.PILToTensor(),  # using PILToTensor() to load zoe depth unscaled
        transforms.Resize(192)
    ])

    train_data_dir = '/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_30scenes_newPitch_originalModel'
    test_data_dir = '/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_2scenes_newPitch_3diffModel'


    valid_dataset = DepthDataset_preload(data_dir=test_data_dir, split='', transform=transform, target_transform=target_transform,
                                         pseudo_depth_transform=pseudo_depth_transform, mask_transform=mask_transform, device=device, sub_video_count=10)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8, drop_last=True)

    # train_dataset = DepthDataset_preload(data_dir=train_data_dir, split='', transform=transform, target_transform=target_transform,
                                        #  pseudo_depth_transform=pseudo_depth_transform, mask_transform=mask_transform, device=device, sub_video_count=20,cut_off_scene=5*54) # each scene has 54 videos
    # use a last 20 scenes of total 30 for training
    # frames_for_earch_scene = 13014  # 54 videos

    # train_dataloader = DataLoader(train_dataset, batch_size=64,
    #                               shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=2)

    

    # create segmentation model with pretrained encoder
    CLASSES = ["all_depth"]
    model = smp.Unet(
        encoder_name='resnet18',
        encoder_weights='imagenet',
        in_channels=4,
        classes=len(CLASSES),
        activation='ReLU',  # could be None for logits or 'softmax2d' for multiclass segmentation
    )

    model.to(device)
    # define loss function
    # define optimizer
    # define learning rate scheduler (not used )
    
    search_space = {
    "lr": tune.loguniform(8, 1),
    "lambd": tune.uniform(0.1, 0.9)
    # "batch_size": tune.choice([16,32,64])
    }

# train(model, train_data_dir, valid_dataloader, 1000, run_name, optimizer=optimizer, lr_scheduler,transform=transform,
#           target_transform=target_transform, pseudo_depth_transform=pseudo_depth_transform, mask_transform=mask_transform, metric=metric, writer=writer),


    tuner = tune.Tuner(
    tune.with_parameters(train, model=model, train_data_dir=train_data_dir, valid_dataloader=valid_dataloader, num_epochs=500, run_name=run_name, transform=transform,
          target_transform=target_transform, pseudo_depth_transform=pseudo_depth_transform, mask_transform=mask_transform),
    run_config=air.RunConfig(local_dir="./ray_results", name="test_experiment"),
    tune_config=tune.TuneConfig(
        num_samples=1,
        max_concurrent_trials = 2,
        scheduler=ASHAScheduler(metric="object_val_loss", mode="min"),
    ),
    param_space=search_space,
    )


    results = tuner.fit()

    # Obtain a trial dataframe from all run trials of this `tune.run` call.
    dfs = {result.log_dir: result.metrics_dataframe for result in results}
    # Plot by epoch
    # ax = None  # This plots everything on the same plot
    # for d in dfs.values():
    #     ax = d.object_val_loss.plot(ax=ax, legend=False)


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
