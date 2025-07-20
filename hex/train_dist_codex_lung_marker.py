import os
from os.path import join
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import tqdm
from sklearn.metrics import mean_squared_error

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torch.cuda.amp import autocast

from scipy.stats import pearsonr
import robust_loss_pytorch
from hex.hex_architecture import CustomModel

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    print(f"Successfully initialized process group. "
          f"Rank: {dist.get_rank()}, "
          f"World Size: {dist.get_world_size()}, "
          f"Master Port: {os.environ.get('MASTER_PORT', 'Not set')}")


def cleanup():
    dist.destroy_process_group()



def main():
    setup()

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")

    seed_torch(global_rank)

    save_dir = ""
    data_dir = ""
    img_dir = Path(data_dir) / "he_patches"
    csv_dir = Path(data_dir) / "results"
    csv_files = list(csv_dir.glob('*.csv'))

    patient_ids = [csv_file.stem.split('.')[0] for csv_file in csv_files]

    #example split for test
    np.random.shuffle(patient_ids)
    train_ids = patient_ids[:int(len(patient_ids) * 0.8)]
    val_ids = patient_ids[int(len(patient_ids) * 0.8):]

    train_csvs = []
    for train_id in train_ids:
        train_csv = pd.read_csv(join(csv_dir, f'{train_id}.csv'))
        train_csv['patch_id'] = train_csv['slide'].astype(str) + '_' + train_csv['index'].astype(str)
        train_csvs.append(train_csv)
    train_csvs = pd.concat(train_csvs)
    train_csvs.reset_index(drop=True, inplace=True)
    train_csvs['images'] = str(img_dir) + '/' + train_csvs['slide'].astype(str) + '/' + train_csvs['slide'].astype(
        str) + '_' + train_csvs['index'].astype(str) + '.png'

    val_csvs = []
    for val_id in val_ids:
        val_csv = pd.read_csv(join(csv_dir, f'{val_id}.csv'))
        val_csv['patch_id'] = val_csv['slide'].astype(str) + '_' + val_csv['index'].astype(str)
        val_csvs.append(val_csv)
    val_csvs = pd.concat(val_csvs)
    val_csvs.reset_index(drop=True, inplace=True)
    val_csvs['images'] = str(img_dir) + '/' + val_csvs['slide'].astype(str) + '/' + val_csvs['slide'].astype(
        str) + '_' + val_csvs['index'].astype(str) + '.png'


    label_columns = [f'mean_intensity_channel{i}' for i in range(1, 41)]
    biomarker_names = {
        1: "DAPI",
        2: "CD8",
        3: "Pan-Cytokeratin",
        4: "CD3e",
        5: "CD163",
        6: "CD20",
        7: "CD4",
        8: "FAP",
        9: "CD138",
        10: "CD11c",
        11: "CD66b",
        12: "aSMA",
        13: "CD68",
        14: "Ki67",
        15: "CD31",
        16: "Collagen IV",
        17: "Granzyme B",
        18: "MMP9",
        19: "PD-1",
        20: "CD44",
        21: "PD-L1",
        22: "E-cadherin",
        23: "LAG3",
        24: "Mac2/Galectin-3",
        25: "FOXP3",
        26: "CD14",
        27: "EpCAM",
        28: "CD21",
        29: "CD45",
        30: "MPO",
        31: "TCF-1",
        32: "ICOS",
        33: "Bcl-2",
        34: "HLA-E",
        35: "CD45RO",
        36: "VISTA",
        37: "HIF1A",
        38: "CD39",
        39: "CD40",
        40: "HLA-DR"
    }
    train_csvs.reset_index(drop=True, inplace=True)
    val_csvs.reset_index(drop=True, inplace=True)

    img_size = 384
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])

    train_dataset = PatchDataset(train_csvs, label_columns, transform_train)
    val_dataset = PatchDataset(val_csvs, label_columns, transform_val)


    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank)

    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=48, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=48, sampler=val_sampler, num_workers=num_workers)


    num_outputs = len(label_columns)
    model = CustomModel(visual_output_dim=1024, num_outputs=num_outputs).to(device)
    pretrained = False  # Set to True if you want to load your pretrained weights or the provided demo checkpoint
    if pretrained:
        # Load the saved weights
        checkpoint_path = "./sample_checkpoints.pth"
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model weights from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
    model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)


    # Freeze all parameters
    for param in model.module.parameters():
        param.requires_grad = False
    # Unfreeze the last 4 encoder layers
    for layer in model.module.visual.beit3.encoder.layers[-4:]:
        for param in layer.parameters():
            param.requires_grad = True
    # Unfreeze the final layer norm
    for param in model.module.visual.beit3.encoder.layer_norm.parameters():
        param.requires_grad = True
    # Unfreeze the regression head
    for param in model.module.regression_head.parameters():
        param.requires_grad = True
    for param in model.module.regression_head1.parameters():
        param.requires_grad = True

    if global_rank == 0:
        print_network(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    criterion_ad = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        num_dims=40, float_dtype=torch.float32, device=local_rank)
    optimizer.add_param_group({'params': criterion_ad.parameters(), 'lr': 1e-5, 'name': 'criterion_ad'})
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    num_epochs = 200
    losses = []
    epoch_losses = []
    val_losses = []

    if global_rank == 0:
        writer_dir = join(save_dir, "runs")
        if not os.path.isdir(writer_dir):
            os.makedirs(writer_dir)
        writer = SummaryWriter(writer_dir)

    checkpoint_dir = join(save_dir, 'checkpoints')
    if global_rank == 0 and not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_loss = float('inf')
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        model.module.training_status = True
        running_loss = 0.0
        all_preds = []
        all_labels = []
        encodings= []
        train_loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), disable=(global_rank != 0))
        for i, data in train_loop:
            inputs, labels = data[0].to(device, dtype=torch.float16), data[1].to(device, dtype=torch.float16)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                optimizer.zero_grad()
                outputs,feature = model(inputs,labels,epoch)
                loss = torch.mean(criterion_ad.lossfun(outputs.to(device, dtype=torch.float32) - labels.to(device, dtype=torch.float32)))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            encodings.extend(feature.data.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.detach().cpu().numpy())
        if epoch >= model.module.FDS.start_update:
            encodings, all_labels = torch.from_numpy(np.vstack(encodings)).to(device), torch.from_numpy(np.vstack(all_labels)).to(device)
            model.module.FDS.update_last_epoch_stats(epoch)
            model.module.FDS.update_running_stats(encodings, all_labels.cpu().numpy(), epoch)
        avg_loss = torch.tensor(running_loss / len(train_loader), device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss / world_size

        if global_rank == 0:
            writer.add_scalar('Loss/train', avg_loss.item(), epoch + 1)
            mse_per_output = np.nanmean((np.array(all_labels) - np.array(all_preds)) ** 2, axis=0)
            for j in range(num_outputs):
                writer.add_scalar(f'MSE_train/{biomarker_names[j+1]}', mse_per_output[j], epoch + 1)
            avg_train_mse = np.nanmean(mse_per_output)
            writer.add_scalar('MSE_train/avg', avg_train_mse, epoch + 1)


        model.eval()
        model.module.training_status = False
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            val_loop = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), disable=(global_rank != 0))
            for i, data in val_loop:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs,_ = model(inputs,labels,epoch)
                all_labels.append(labels)
                all_preds.append(outputs)

        # Concatenate all tensors
        all_labels = torch.cat(all_labels, dim=0)
        all_preds = torch.cat(all_preds, dim=0)

        # Gather results from all processes
        gathered_labels = [torch.zeros_like(all_labels) for _ in range(world_size)]
        gathered_preds = [torch.zeros_like(all_preds) for _ in range(world_size)]

        dist.all_gather(gathered_labels, all_labels)
        dist.all_gather(gathered_preds, all_preds)

        # Concatenate results on rank 0
        if global_rank == 0:
            all_labels = torch.cat(gathered_labels).cpu().numpy()
            all_preds = torch.cat(gathered_preds).cpu().numpy()

            # Calculate MSE for each biomarker
            mse_per_biomarker = np.nanmean((all_labels - all_preds)**2, axis=0)
            # Calculate overall MSE
            overall_mse = np.nanmean((all_labels - all_preds) ** 2)

            # Calculate Pearson R for each biomarker
            pearson_r_per_biomarker = []
            for i in range(all_labels.shape[1]):  # For each biomarker
                r, _ = pearsonr(all_labels[:, i], all_preds[:, i])
                pearson_r_per_biomarker.append(r)

            # Calculate average Pearson R across all biomarkers
            avg_pearson_r = np.nanmean(pearson_r_per_biomarker)


        if global_rank == 0:
            writer.add_scalar('MSE_val/avg', overall_mse, epoch + 1)
            writer.add_scalar('Pearson_R_val/avg', avg_pearson_r, epoch + 1)

            for i in range(len(mse_per_biomarker)):
                writer.add_scalar(f'MSE_val/{biomarker_names[i+1]}', mse_per_biomarker[i], epoch + 1)
                writer.add_scalar(f'Pearson_R_val/{biomarker_names[i+1]}', pearson_r_per_biomarker[i], epoch + 1)

            print(f"Epoch {epoch+1}")
            print(f"Average MSE: {overall_mse:.4f}")
            print(f"Average Pearson R: {avg_pearson_r:.4f}")

        save_frequency = 1  # Save every 5 epochs
        if (epoch + 1) % save_frequency == 0:
            dist.barrier()
            if global_rank == 0:
                torch.save(model.module.state_dict(), join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
                print(f"Model weights saved for epoch {epoch + 1}")
            dist.barrier()
    if global_rank == 0:
        print("Finished Training")

    cleanup()


if __name__ == "__main__":
    main()
