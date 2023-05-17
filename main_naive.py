import os

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import VGG16_Weights
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DepthDataset(Dataset):
    def __init__(self, data_dir, split='', transform=None, target_transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.scene_names = sorted(os.listdir(os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p')))
        self.data = []

        for scene_name in self.scene_names:
            rgb_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/JPEGImages/480p', scene_name)
            depth_folder = os.path.join(data_dir, split, 'habitat_sim_DAVIS/Annotations/480p_depth', scene_name)
            pseudo_depth_folder = os.path.join(data_dir, split, 'zoe_depth_raw', scene_name)
            for filename in os.listdir(rgb_folder):
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    rgb_path = os.path.join(rgb_folder, filename)
                    depth_path = os.path.join(depth_folder, filename[:-4]+'.png')
                    pseudo_depth_path = os.path.join(pseudo_depth_folder, filename[:-4]+'.png')
                    self.data.append((rgb_path, depth_path, pseudo_depth_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rgb_path, depth_path, pseudo_depth_path = self.data[index]

        # Load RGB image
        rgb_image = Image.open(rgb_path).convert('RGB')
        depth_image = Image.open(depth_path)
        if self.transform:
            rgb_image = self.transform(rgb_image)
        if self.target_transform:
            depth_image = self.target_transform(depth_image)

        # Load depth maps
        ground_truth_depth = depth_image
        # rescale since the raw pseudo depth is saved as 16-bit png
        pseudo_depth = np.array(Image.open(pseudo_depth_path)) / 256.
        # pseudo_depth = np.stack((pseudo_depth,)*3, axis=0)
        pseudo_depth = torch.from_numpy(pseudo_depth).float()

        # Combine RGB and depth maps into a single input tensor
        input_tensor = torch.cat((rgb_image, pseudo_depth.unsqueeze(dim=0)), dim=0) # type: ignore
        # rgb_image, ground_truth_depth,pseudo_depth = rgb_image.to(device), ground_truth_depth.to(device),pseudo_depth.to(device)
        input_tensor = input_tensor.to(device)
        ground_truth_depth = ground_truth_depth.to(device)
        # return rgb_image, pseudo_depth, ground_truth_depth
        return input_tensor, ground_truth_depth


class DepthScaleShiftNet(nn.Module):
    def __init__(self):
        super(DepthScaleShiftNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # Load VGG-16 as the backbone
        self.backbone = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        # Modify the first convolution layer to accept 4 input channels
        self.backbone[0] = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1) # type: ignore
        # Replace the last max pooling layer with a convolutional layer to output detph map
        self.backbone[-1] = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1) # type: ignore
        # TODO: use upconv or unet if performance is bad
        # TODO: https://arxiv.org/abs/2006.11339

        # # Scale and shift prediction layers
        # self.scale_prediction = nn.Sequential(
        #     nn.Linear(512 * 8 * 8, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1)
        # )
        # self.shift_prediction = nn.Sequential(
        #     nn.Linear(512 * 8 * 8, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1)
        # )

        # # to output a single scale + shift
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 1000),
        # )

    def forward(self, x):
        # # Pass input tensor through the backbone
        # features = self.backbone(input_tensor)

        # # Flatten features and pass them through the scale and shift prediction layers
        # x = features.view(features.size(0), -1)
        # scale = self.scale_prediction(x)
        # shift = self.shift_prediction(x)

        # return scale, shift
        x = self.backbone(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x


def train(train_dataloader, test_dataloader, num_epochs,criterion, save_period=10):
    writer = SummaryWriter()
    # TODO: eval, train-eval curve, checkpoint on best
    model = DepthScaleShiftNet()
    model.to(device)
    print(f"model structure: \n {model}")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # model.set(train)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()

            scale_shift = model(inputs)
            # print(scale_shift[:,[0]].unsqueeze(dim=1).shape)
            # inputs_pseudo_dpeth = inputs[:,3,:,:]
            loss = criterion(scale_shift, labels)

            loss.backward(loss)
            optimizer.step()

            # Update running loss
            running_loss += loss.item() * inputs.size(0)
        
        # Compute epoch loss
        epoch_loss = running_loss / len(train_dataset)

        # save loss info to tensorboard
        writer.add_scalar('Loss/train', epoch_loss, epoch)


        # save checkpoint every save_period
        if epoch > 0 and save_period > 0 and epoch % save_period == 0:
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, '/home/gao/dev/project_remote/zoe_refine/checkpoint/'+f"{epoch:04d}.pth")

        # print loss info

        # eval
        model.eval()
        with torch.no_grad():
            eval_running_loss = 0.0
            for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
                # TODO: 一个直接生成差值的depth_img, 直接用相加来算loss
                scale_shift = model(inputs)
                # print(scale_shift[:,[0]].unsqueeze(dim=1).shape)
                inputs_pseudo_dpeth = inputs[:,3,:,:]
                # TODO: loss 只计算object 的 mask
                loss = criterion(scale_shift[:,[0]].unsqueeze(dim=1) * inputs_pseudo_dpeth + scale_shift[:,[1]].unsqueeze(dim=1), labels[:,0,:,:])
                eval_running_loss += loss.item() * inputs.size(0)

            test_epoch_loss = eval_running_loss / len(test_dataloader)
            writer.add_scalar('Loss/test',  test_epoch_loss, epoch)
            print(f"Epoch {epoch+1} train loss: {epoch_loss:.4f}")
            print(f"Epoch {epoch+1} eval loss: {test_epoch_loss:.4f}")

        writer.close()

def test():
    pass


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t/255.*10.)  # Scale habitatDyn depth pixel values to depth in meters
    ])

    # TODO: 先用不同的2scene做个test
    # TODO: 用30个scene的后20个做训练
    train_data_dir = '/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_5scenes_newPitch_original'
    test_data_dir = '/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_2scenes_newPitch_diffPath'

    train_dataset = DepthDataset(data_dir=train_data_dir, split='', transform=transform, target_transform=target_transform)
    # TODO: only use a subset to train for now
    sub_train_dataset = torch.utils.data.Subset(train_dataset, range(0, 1024))
    train_dataloader = DataLoader(sub_train_dataset, batch_size=16, shuffle=True)

    test_dataset = DepthDataset(data_dir=test_data_dir, split='', transform=transform, target_transform=target_transform)
    # only use a subset of test_dataset to eval
    sub_test_dataset = torch.utils.data.Subset(test_dataset, range(0, 256))
    test_dataloader = DataLoader(sub_test_dataset, batch_size=16, shuffle=True)

    criterion = nn.MSELoss()

    train(train_dataloader, test_dataloader, 100, criterion)