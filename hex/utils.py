
import os
import random
import torch
import torch.nn as nn
import numpy as np
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import logging

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class PatchDataset(Dataset):
    def __init__(self, csv,label_columns, transform=None):
        self.images = csv['images'].values
        self.labels = csv[label_columns].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx, :]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label, image_path

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    print("\nTrainable parameters:")
    for name, param in net.named_parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
            print(f"{name}, Shape: {param.shape}")

    print('\nTotal number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)