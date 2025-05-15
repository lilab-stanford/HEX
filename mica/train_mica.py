from __future__ import print_function

import argparse
import pdb
import os
import math
import sys
from timeit import default_timer as timer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

### Internal Imports
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from file_utils import save_pkl, load_pkl
from core_utils import train
from utils import get_custom_exp_code

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler


def main(args):
    #### Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    latest_val_cindex = []
    folds = np.arange(start, end)

    ### Start 5-Fold CV Evaluation.
    for i in folds:
        start_time = timer()
        seed_torch(args.seed)
        results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))


        ### Gets the Train + Val Dataset Loader.
        train_dataset, val_dataset = dataset.return_splits(from_id=False,csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)


        ### Run Train-Val on Survival Task.
        if args.task_type == 'survival':
            val_latest, cindex_all_i = train(datasets, i, args)
            latest_val_cindex.append(cindex_all_i)

        ### Write Results for Each Split to PKL
        save_pkl(results_pkl_path, val_latest)
        end_time = timer()
        print('Fold %d Time: %f seconds' % (i, end_time - start_time))

        ### Finish 5-Fold CV Evaluation.
        final_df1 = pd.DataFrame(latest_val_cindex)
        final_df1['folds'] = final_df1.index
        final_df1['folds'] = final_df1['folds'] + start
        final_df = pd.DataFrame({'folds': np.arange(start, i + 1)})
        final_df = pd.merge(final_df, final_df1, how='left', on=['folds'])
        col_mean = pd.DataFrame(final_df.mean(axis=0)).T
        final_df = pd.concat([final_df, col_mean])
        final_df.reset_index(drop=True, inplace=True)

        final_df.to_csv(os.path.join(args.results_dir, f'summary_latest_{start}_{end}.csv'))


### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir', type=str, default='path/to/data_root_dir',
                    help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir', type=str, default='./results', help='Results directory (Default: ./results)')
parser.add_argument('--which_splits', type=str, default='5foldcv',
                    help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir', type=str, default='tcga_blca_100',
                    help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
parser.add_argument('--log_data', action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Whether or not to overwrite experiments (if already ran)')

### Model Parameters.
parser.add_argument('--model_type', type=str, choices=['snn', 'deepset', 'amil', 'mi_fcn', 'mcat'], default='mcat',
                    help='Type of model (Default: mcat)')
parser.add_argument('--mode', type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'], default='coattn',
                    help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion', type=str, choices=['None', 'concat', 'bilinear'], default='concat',
                    help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig', action='store_true', default=False,
                    help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats', action='store_true', default=False,
                    help='Use genomic features as tabular features.')
parser.add_argument('--drop_out', action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi', type=str, default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc', type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'],
                    default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg', type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv', type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'], default='None',
                    help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='Enable weighted sampling')
parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            # 'inst_loss': args.inst_loss,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size_wsi': args.model_size_wsi,
            'model_size_omic': args.model_size_omic,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt}
print('\nLoad Dataset')

if 'survival' in args.task:
    args.n_classes = 4
    dataset = Generic_MIL_Survival_Dataset(csv_path=args.csv_path,
                                           codex_deep=args.codex_deep,
                                           mode=args.mode,
                                           apply_sig=args.apply_sig,
                                           data_dir=args.data_dir,
                                           shuffle=False,
                                           seed=args.seed,
                                           print_info=True,
                                           patient_strat=False,
                                           n_bins=args.n_classes,
                                           label_col='survival_months',
                                           ignore=[])
else:
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path=args.csv_path,
                                  codex_deep=args.codex_deep,
                                  mode=args.mode,
                                  apply_sig=args.apply_sig,
                                  data_dir=args.data_dir,
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  patient_strat=False,
                                  label_col='label',
                                  ignore=[])

if isinstance(dataset, Generic_MIL_Survival_Dataset):
    args.task_type = 'survival'
else:
    args.task_type = 'class'

### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

### Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, str(args.exp_code))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)


settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
