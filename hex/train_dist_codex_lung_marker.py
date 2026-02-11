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
from utils import *


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

    save_dir = "./results"
    data_dir = "./sample_data/"
    img_dir = Path(data_dir) / "he_patches"
    csv_dir = Path(data_dir) / "channel_registered"
    csv_files = list(csv_dir.glob('*.csv'))

    patient_ids = [csv_file.stem.split('.')[0] for csv_file in csv_files]

    split_csv = Path(data_dir) / "splits_0.csv"
    if split_csv.exists():
        split_df = pd.read_csv(split_csv)
        train_ids = [str(int(x)) for x in split_df["train"].dropna().tolist()]
        val_ids = [str(int(x)) for x in split_df["val"].dropna().tolist()]
    else:
        np.random.shuffle(patient_ids)
        train_ids = patient_ids[:int(len(patient_ids) * 0.8)]
        val_ids = patient_ids[int(len(patient_ids) * 0.8):]

    if dist.is_initialized() and dist.get_world_size() > 1:
        obj_list = [train_ids, val_ids]
        dist.broadcast_object_list(obj_list, src=0)
        train_ids, val_ids = obj_list
    overlap = set(train_ids) & set(val_ids)
    if len(overlap) != 0:
        raise ValueError(f"Train/val patient_id overlap: {sorted(list(overlap))[:20]}")
    train_csvs = []
    for train_id in train_ids:
        train_csv = pd.read_csv(join(csv_dir, f'{train_id}.csv'))
        train_csv['patch_id'] = train_csv['slide'].astype(str) + '_' + train_csv['index'].astype(str)
        train_csvs.append(train_csv)
    train_csvs = pd.concat(train_csvs)
    train_csvs.reset_index(drop=True, inplace=True)
    label_columns = [f'mean_intensity_channel{i}' for i in range(1, 41)]
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
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)

    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=48, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=48, sampler=val_sampler, num_workers=num_workers)


    num_outputs = len(label_columns)

    # FDS biomarker indices (empty disables). Optionally use a small, empirically chosen set.
    # Examples: [] , [0] , list(range(num_outputs)) , [0, 3, 7]
    FDS_ACTIVE_MARKERS = list(range(num_outputs))
    FDS_OFF_EPOCH = 50

    model = CustomModel(visual_output_dim=1024, num_outputs=num_outputs, fds_active_markers=FDS_ACTIVE_MARKERS).to(device)
    pretrained = False  # Set to True if you want to load your pretrained weights
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

    lr = 1e-5
    lr_gamma = 0.95

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion_ad = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        num_dims=num_outputs, float_dtype=torch.float32, device=device)

    if dist.is_initialized() and dist.get_world_size() > 1:
        for p in criterion_ad.parameters():
            dist.broadcast(p.data, src=0)

    optimizer.add_param_group({'params': criterion_ad.parameters(), 'lr': lr, 'name': 'criterion_ad'})

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    num_epochs = 120

    losses = []
    epoch_losses = []
    val_losses = []

    if global_rank == 0:
        writer_dir = join(save_dir, "runs/test")
        if not os.path.isdir(writer_dir):
            os.makedirs(writer_dir)
        writer = SummaryWriter(writer_dir)

    checkpoint_dir = join(save_dir, 'checkpoints/test')
    if global_rank == 0 and not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_loss = float('inf')
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        use_fds_train = (len(FDS_ACTIVE_MARKERS) > 0) and (epoch < FDS_OFF_EPOCH)
        model.module.training_status = use_fds_train

        if epoch >= 100:
            if epoch == 100:
                for p in model.module.visual.parameters():
                    p.requires_grad = False
            model.module.visual.eval()

        running_loss = 0.0

        mse_sum = torch.zeros(num_outputs, device=device, dtype=torch.float64)
        mse_count = torch.zeros(1, device=device, dtype=torch.long)

        # FDS for active biomarkers
        if use_fds_train and (epoch >= model.module.FDS[0].start_update):
            _bucket_num = int(model.module.FDS[0].bucket_num)
            _feat_dim = int(model.module.FDS[0].feature_dim)
            _active = list(FDS_ACTIVE_MARKERS)
            fds_count = torch.zeros(len(_active), _bucket_num, device=device, dtype=torch.long)
            fds_sum = torch.zeros(len(_active), _bucket_num, _feat_dim, device=device, dtype=torch.float32)
            fds_sumsq = torch.zeros_like(fds_sum)
        train_loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), disable=(global_rank != 0))
        for i, data in train_loop:
            inputs, labels = data[0].to(device, dtype=torch.float16), data[1].to(device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                optimizer.zero_grad()
                outputs,feature = model(inputs,labels,epoch)

                loss = torch.mean(criterion_ad.lossfun(outputs.to(device, dtype=torch.float32) - labels.to(device, dtype=torch.float32)))

            scaler.scale(loss).backward()

            # --- Sync grads ---
            scaler.unscale_(optimizer)
            if dist.is_initialized() and dist.get_world_size() > 1:
                ws = dist.get_world_size()
                for p in criterion_ad.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(ws)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            diff = (outputs.to(torch.float32) - labels.to(torch.float32))
            mse_sum += (diff * diff).sum(dim=0).to(torch.float64)
            mse_count += labels.size(0)

            if use_fds_train and (epoch >= model.module.FDS[0].start_update):
                x = feature.detach().to(torch.float32)  # [bsz, feat_dim]
                for _k, _j in enumerate(_active):
                    y = labels[:, _j].detach().to(torch.float32).clamp_(0.0, 1.0)
                    idx = torch.clamp((y * _bucket_num).to(torch.long), 0, _bucket_num - 1)
                    fds_count[_k].index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
                    fds_sum[_k].index_add_(0, idx, x)
                    fds_sumsq[_k].index_add_(0, idx, x * x)
        if use_fds_train and (epoch >= model.module.FDS[0].start_update):
            dist.all_reduce(fds_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(fds_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(fds_sumsq, op=dist.ReduceOp.SUM)

            if len(FDS_ACTIVE_MARKERS) == 1:
                j0 = int(FDS_ACTIVE_MARKERS[0])
                model.module.FDS[j0].update_running_stats_from_moments(fds_count[0], fds_sum[0], fds_sumsq[0], epoch)
                model.module.FDS[j0].update_last_epoch_stats(epoch + 1)
            else:
                for _k, _j in enumerate(_active):
                    model.module.FDS[_j].update_running_stats_from_moments(fds_count[_k], fds_sum[_k], fds_sumsq[_k], epoch)
                    model.module.FDS[_j].update_last_epoch_stats(epoch + 1)
        avg_loss = torch.tensor(running_loss / len(train_loader), device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss / world_size

        dist.all_reduce(mse_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mse_count, op=dist.ReduceOp.SUM)

        if global_rank == 0:
            writer.add_scalar('Loss/train', avg_loss.item(), epoch + 1)
            mse_per_output = (mse_sum / mse_count.clamp_min(1)).detach().cpu().numpy()
            for j in range(num_outputs):
                writer.add_scalar(f'MSE_train/{biomarker_names[j+1]}', float(mse_per_output[j]), epoch + 1)
            avg_train_mse = float(np.nanmean(mse_per_output))
            writer.add_scalar('MSE_train/avg', avg_train_mse, epoch + 1)


        model.eval()
        model.module.training_status = False
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            val_loop = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), disable=(global_rank != 0))
            for i, data in val_loop:
                inputs, labels = data[0].to(device, dtype=torch.float16), data[1].to(device, dtype=torch.float32)
                outputs,_ = model(inputs,labels,epoch)
                all_labels.append(labels)
                all_preds.append(outputs)

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

            mse_per_biomarker = np.nanmean((all_labels - all_preds)**2, axis=0)
            overall_mse = np.nanmean((all_labels - all_preds) ** 2)

            pearson_r_per_biomarker = []
            for i in range(all_labels.shape[1]):
                r, _ = pearsonr(all_labels[:, i], all_preds[:, i])
                pearson_r_per_biomarker.append(r)

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
        scheduler.step()
        save_frequency = 5  # Save every 5 epochs
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