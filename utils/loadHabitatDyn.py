import os

import numpy as np
from PIL import Image

def load_HabitatDyn_img(dataset, idx):
    rgb_path, pseudo_depth_path, depth_path, mask_path = dataset[idx]

    #load RGB image
    rgb_image = Image.open(rgb_path).convert('RGB').copy()
    depth_image = Image.open(depth_path).copy()
    pseudo_depth = Image.open(pseudo_depth_path).copy()
    mask = Image.open(mask_path).copy()

    return rgb_image, pseudo_depth, depth_image, mask