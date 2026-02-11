
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
from timm import create_model
from musk import utils, modeling
from scipy.ndimage import gaussian_filter1d,convolve1d
from scipy.stats import gaussian_kde
from scipy.interpolate import interpn
from scipy.signal import convolve
from scipy.signal.windows import triang

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


class CustomModel(nn.Module):
    def __init__(self, visual_output_dim, num_outputs, fds_active_markers=None):
        super(CustomModel, self).__init__()
        self.fds_active_markers = [0] if fds_active_markers is None else list(fds_active_markers)
        model_config = "musk_large_patch16_384"
        model_musk = create_model(model_config, vocab_size=64010)
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model_musk, 'model|module', '')
        self.visual = model_musk
        self.regression_head = nn.Sequential(
            nn.Linear(visual_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.regression_head1 = nn.Sequential(
            nn.Linear(128, num_outputs),
        )
        config = dict(feature_dim=128, start_update=0, start_smooth=10, kernel='gaussian', ks=9, sigma=2)
        self.FDS = nn.ModuleList([FDS(**config) for _ in range(num_outputs)])
        self.training_status = True

    def forward(self, x,labels,epoch):
        x = self.visual(
            image=x,
            with_head=False,
            out_norm=False
        )[0]
        features = self.regression_head(x)
        if self.training_status and epoch >= self.FDS[0].start_smooth:
            linear = self.regression_head1[0]
            weight = linear.weight
            bias = linear.bias

            # smooth features
            if len(self.fds_active_markers) == 1:
                j0 = int(self.fds_active_markers[0])
                h = self.FDS[j0].smooth(features.clone(), labels[:, j0].cpu(), epoch)
                preds = self.regression_head1(h)
            else:
                preds = self.regression_head1(features)
                for j in self.fds_active_markers:
                    j = int(j)
                    h = self.FDS[j].smooth(features.clone(), labels[:, j].cpu(), epoch)
                    preds[:, j] = F.linear(h, weight[j:j + 1], None if bias is None else bias[j:j + 1]).squeeze(1)
        else:
            preds = self.regression_head1(features)
        return preds, features

class PatchDataset(Dataset):
    def __init__(self, csv,label_columns, transform=None):
        self.images = csv['images'].values
        self.labels = csv[label_columns].values.astype(np.float32, copy=False)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = torch.from_numpy(self.labels[idx, :])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label, image_path

def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10.0):
    if torch.sum(v1) < 1e-10:
        return matrix

    v1p = torch.clamp(v1, min=0.0)
    v2p = torch.clamp(v2, min=0.0)

    if (v1p == 0.).any():
        valid = (v1p != 0.)
        if valid.any():
            factor = torch.clamp(v2p[valid] / v1p[valid], clip_min, clip_max)
            matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2p / v1p, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2

class FDS(nn.Module):

    def __init__(self, feature_dim, bucket_num=50, bucket_start=0, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=9, sigma=2, momentum=0.9):
        super(FDS, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.half_ks = (ks - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth

        self.register_buffer('epoch', torch.zeros(1, dtype=torch.long).fill_(start_update))
        self.register_buffer('running_mean', torch.zeros(bucket_num - bucket_start, feature_dim, dtype=torch.float32))
        self.register_buffer('running_var', torch.ones(bucket_num - bucket_start, feature_dim, dtype=torch.float32))
        self.register_buffer('running_mean_last_epoch',
                             torch.zeros(bucket_num - bucket_start, feature_dim, dtype=torch.float32))
        self.register_buffer('running_var_last_epoch',
                             torch.ones(bucket_num - bucket_start, feature_dim, dtype=torch.float32))
        self.register_buffer('smoothed_mean_last_epoch',
                             torch.zeros(bucket_num - bucket_start, feature_dim, dtype=torch.float32))
        self.register_buffer('smoothed_var_last_epoch',
                             torch.ones(bucket_num - bucket_start, feature_dim, dtype=torch.float32))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num - bucket_start, dtype=torch.float32))

        self.register_buffer('kernel_window', self._get_kernel_window(kernel, ks, sigma))
    @staticmethod
    def _get_kernel_window(kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks) / sum(triang(ks))
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = np.array(list(map(laplace, np.arange(-half_ks, half_ks + 1)))) / sum(
                map(laplace, np.arange(-half_ks, half_ks + 1)))

        return torch.tensor(kernel_window, dtype=torch.float32)

    def _get_bucket_idx(self, label):
        label = float(label)
        if label < 0.0:
            label = 0.0
        elif label > 1.0:
            label = 1.0
        idx = int(label * self.bucket_num)
        if idx >= self.bucket_num:
            idx = self.bucket_num - 1
        if idx < self.bucket_start:
            idx = self.bucket_start
        return idx

    def _get_bucket_idx_vec(self, labels):
        labels = np.asarray(labels, dtype=np.float32)
        labels = np.clip(labels, 0.0, 1.0)
        buckets = (labels * self.bucket_num).astype(np.int64)
        buckets = np.clip(buckets, 0, self.bucket_num - 1)
        if self.bucket_start > 0:
            buckets = np.maximum(buckets, self.bucket_start)
        return buckets
    def _update_last_epoch_stats(self):
        self.running_mean_last_epoch.copy_(self.running_mean)
        self.running_var_last_epoch.copy_(self.running_var)

        smoothed_mean = F.conv1d(
            input=F.pad(self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

        smoothed_var = F.conv1d(
            input=F.pad(self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

        self.smoothed_mean_last_epoch.copy_(smoothed_mean)
        self.smoothed_var_last_epoch.copy_(smoothed_var)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1)
        self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch):
        if epoch == int(self.epoch.item()) + 1:
            self.epoch += 1
            self._update_last_epoch_stats()

    def update_running_stats(self, features, labels, epoch):
        if epoch < int(self.epoch.item()):
            return

        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size, "Dimensions of features and labels are not aligned!"

        buckets = self._get_bucket_idx_vec(labels)
        for bucket in np.unique(buckets):
            curr_feats = features[torch.as_tensor((buckets == bucket).astype(bool), device=features.device)]
            curr_num_sample = curr_feats.size(0)
            curr_mean = torch.mean(curr_feats, 0)
            curr_var = torch.var(curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False)

            self.num_samples_tracked[bucket - self.bucket_start] += curr_num_sample
            factor = self.momentum if self.momentum is not None else \
                (1 - curr_num_sample / float(self.num_samples_tracked[bucket - self.bucket_start]))
            factor = 0 if epoch == self.start_update else factor
            self.running_mean[bucket - self.bucket_start] = \
                (1 - factor) * curr_mean + factor * self.running_mean[bucket - self.bucket_start]
            self.running_var[bucket - self.bucket_start] = \
                (1 - factor) * curr_var + factor * self.running_var[bucket - self.bucket_start]

        for bucket in range(self.bucket_start, self.bucket_num):
            if bucket not in np.unique(buckets):
                if float(self.num_samples_tracked[bucket - self.bucket_start].item()) > 0.0:
                    continue
                if bucket == self.bucket_start:
                    self.running_mean[0] = self.running_mean[1]
                    self.running_var[0] = self.running_var[1]
                elif bucket == self.bucket_num - 1:
                    self.running_mean[bucket - self.bucket_start] = self.running_mean[bucket - self.bucket_start - 1]
                    self.running_var[bucket - self.bucket_start] = self.running_var[bucket - self.bucket_start - 1]
                else:
                    self.running_mean[bucket - self.bucket_start] = (self.running_mean[bucket - self.bucket_start - 1] +
                                                                     self.running_mean[bucket - self.bucket_start + 1]) / 2.
                    self.running_var[bucket - self.bucket_start] = (self.running_var[bucket - self.bucket_start - 1] +
                                                                    self.running_var[bucket - self.bucket_start + 1]) / 2.

    @torch.no_grad()
    def update_running_stats_from_moments(self, count, sum_feat, sumsq_feat, epoch):
        """Update running (mean, var) using per-bucket moments instead of per-sample features.
        Args:
            count:      Tensor[int64], shape [bucket_num]
            sum_feat:   Tensor[float32], shape [bucket_num, feature_dim]
            sumsq_feat: Tensor[float32], shape [bucket_num, feature_dim]
        """
        if epoch < int(self.epoch.item()):
            return

        b0 = int(self.bucket_start)
        b1 = int(self.bucket_num)
        count = count.to(dtype=torch.long)
        sum_feat = sum_feat.to(dtype=torch.float32)
        sumsq_feat = sumsq_feat.to(dtype=torch.float32)

        for bucket in range(b0, b1):
            n = int(count[bucket].item())
            if n <= 0:
                continue

            curr_mean = sum_feat[bucket] / float(n)
            if n > 1:
                curr_var = (sumsq_feat[bucket] - (sum_feat[bucket] * sum_feat[bucket]) / float(n)) / float(n - 1)
                curr_var = torch.clamp(curr_var, min=0.0)
            else:
                curr_var = torch.zeros_like(curr_mean)

            self.num_samples_tracked[bucket - b0] += float(n)
            factor = 0.0 if epoch == self.start_update else float(self.momentum)

            self.running_mean[bucket - b0] = (1.0 - factor) * curr_mean.to(self.running_mean.dtype) + factor * self.running_mean[bucket - b0]
            self.running_var[bucket - b0] = (1.0 - factor) * curr_var.to(self.running_var.dtype) + factor * self.running_var[bucket - b0]

        # Make up for zero-sample buckets (match original behavior).
        present = set(int(b + b0) for b in torch.nonzero(count[b0:b1] > 0, as_tuple=False).view(-1).cpu().tolist())
        for bucket in range(b0, b1):
            if bucket in present:
                continue
            if float(self.num_samples_tracked[bucket - b0].item()) > 0.0:
                continue

            if bucket == b0:
                self.running_mean[0] = self.running_mean[1]
                self.running_var[0] = self.running_var[1]
            elif bucket == b1 - 1:
                self.running_mean[bucket - b0] = self.running_mean[bucket - b0 - 1]
                self.running_var[bucket - b0] = self.running_var[bucket - b0 - 1]
            else:
                self.running_mean[bucket - b0] = (self.running_mean[bucket - b0 - 1] + self.running_mean[
                    bucket - b0 + 1]) / 2.0
                self.running_var[bucket - b0] = (self.running_var[bucket - b0 - 1] + self.running_var[
                    bucket - b0 + 1]) / 2.0

    def smooth(self, features, labels, epoch):
        if epoch < self.start_smooth:
            return features

        orig_dtype = features.dtype
        feat = features.to(torch.float32)

        buckets = self._get_bucket_idx_vec(labels)
        for bucket in np.unique(buckets):
            mask = torch.as_tensor((buckets == bucket).astype(bool), device=feat.device)
            feat[mask] = calibrate_mean_var(
                feat[mask],
                self.running_mean_last_epoch[bucket - self.bucket_start],
                self.running_var_last_epoch[bucket - self.bucket_start],
                self.smoothed_mean_last_epoch[bucket - self.bucket_start],
                self.smoothed_var_last_epoch[bucket - self.bucket_start]
            )

        return feat.to(orig_dtype)

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