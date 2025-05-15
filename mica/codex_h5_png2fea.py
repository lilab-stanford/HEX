
import os
from os.path import join
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import h5py
import openslide

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

def check_mag(wsi):
    try:
        currMPP = float(wsi.properties['aperio.MPP'])
    except:
        try:
            currMPP = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])
        except:
            currMPP = 0.25
    if currMPP < 0.2:
        mag = 80
    elif currMPP >= 0.2 and currMPP < 0.3:
        mag = 40
    elif currMPP >= 0.4 and currMPP < 0.6:
        mag = 20
    return mag


slide_ext = '.svs'
patch_size=224
he_dir=Path('')
codex_h5_dir=Path('')
save_dir = ''
os.makedirs(save_dir, exist_ok=True)
codex_h5_files = [x for x in codex_h5_dir.glob('*.h5')]
for h5_file in tqdm(codex_h5_files):
    h5_path = str(h5_file)
    he_id=h5_path.split('/')[-1].split('.h5')[0]
    he_path=he_dir/f'{he_id}{slide_ext}'
    wsi = openslide.open_slide(str(he_path))
    mag=check_mag(wsi)
    scale_down_factor = int(patch_size / (40/mag))
    width, height = wsi.dimensions[0]//scale_down_factor+1, wsi.dimensions[1]//scale_down_factor+1
    codex_image = np.zeros((height, width, 40), dtype=np.float16)
    with h5py.File(h5_path, 'r') as f:
        codex_prediction = f['codex_prediction'][:]
        coords = f['coords'][:]
    for i in range(len(coords)):
        x, y = coords[i]
        x, y = int(x/scale_down_factor), int(y/scale_down_factor)
        codex_image[y, x] = codex_prediction[i]
    #save the image
    save_path = join(save_dir, f'{he_id}.npy')
    np.save(save_path, codex_image)

class ImageChannelDataset(Dataset):
    def __init__(self, img_dir, num_channels=40, transform=None):
        self.img_dir = Path(img_dir)
        self.img_paths = list(self.img_dir.glob('*.npy'))
        self.transform = transform
        self.num_channels = num_channels

    def __len__(self):
        return len(self.img_paths) * self.num_channels

    def __getitem__(self, idx):
        img_idx = idx // self.num_channels
        channel_idx = idx % self.num_channels

        img_path = self.img_paths[img_idx]
        img_name = img_path.stem

        # Load the image
        img = np.load(img_path)

        if img.ndim != 3 or img.shape[2] != self.num_channels:
            raise ValueError(f"Expected image shape [H, W, {self.num_channels}], got {img.shape}")

        # Extract the specific channel and stack it
        channel = img[:, :, channel_idx]
        channel_stacked = np.stack([channel, channel, channel], axis=2)

        # Convert to PIL Image
        channel_pil = Image.fromarray(channel_stacked.astype('uint8'))

        if self.transform:
            channel_transformed = self.transform(channel_pil)
        else:
            channel_transformed = channel_pil

        return {
            'image': channel_transformed,
            'name': img_name,
            'channel_idx': channel_idx
        }

img_dir=Path('')
imgs = [x for x in img_dir.glob('*.npy')]
save_dir = Path('')
save_dir.mkdir(exist_ok=True)

transform_val = transforms.Compose([
    transforms.Resize((224,224),interpolation=3),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Create dataset and dataloader
batch_size = 64*8
num_workers = 8
num_channels=40
dataset = ImageChannelDataset(img_dir, num_channels=40, transform=transform_val)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
model.cuda()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
_ = model.eval()

# Create h5 file for saving features
h5_path = save_dir / 'features.h5'
features_dict = {}

with h5py.File(h5_path, 'w') as h5_file:
    for batch in tqdm(dataloader, desc='Extracting features'):
        try:
            images = batch['image'].cuda()
            names = batch['name']
            channel_indices = batch['channel_idx']

            with torch.no_grad():
                features = model(images)

            features = features.cpu().numpy()

            # Organize features by image name
            for idx, (name, channel_idx) in enumerate(zip(names, channel_indices)):
                if name not in features_dict:
                    features_dict[name] = np.zeros((num_channels, features.shape[1]))
                features_dict[name][channel_idx] = features[idx]

        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            continue

    # Save features
    for name, features in features_dict.items():
        h5_file.create_dataset(name, data=features)

print(f"Features saved to {h5_path}")

# Verify the saved features
with h5py.File(h5_path, 'r') as f:
    print("\nVerifying saved features:")
    print(f"Total number of images: {len(f.keys())}")
    first_key = list(f.keys())[0]
    print(f"Feature shape for each image: {f[first_key].shape}")  # Should be [40, feature_dim]

a=1