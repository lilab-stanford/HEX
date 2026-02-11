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
from sksurv.metrics import concordance_index_censored
from captum.attr import IntegratedGradients

### Internal Imports
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
from file_utils import save_pkl, load_pkl
from core_utils import train
from utils import *
from models.model_coattn import MCAT_Surv

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler


def summary_survival(model, loader, n_classes,use_ig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id'].reset_index(drop=True)
    patient_results = {}

    for batch_idx, (data_WSI, codex_feature, label, event_time, c) in enumerate(loader):
        data_WSI = data_WSI.to(device)
        codex_feature = codex_feature.to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)
        label = label.to(device)

        slide_id = slide_ids.iloc[batch_idx]

        hazards, survival, Y_hat, A = model(x_path=data_WSI, x_codex=codex_feature)
        if use_ig:
            model.zero_grad()
            def interpret_patient_mm(x0):
                return model.captum(x_path=x0)
            ig = IntegratedGradients(interpret_patient_mm)
            data_WSI.requires_grad_()
            codex_feature.requires_grad_()
            ig_attr = ig.attribute(data_WSI)
            ig_attr = ig_attr.detach().cpu().numpy()
            ig_attr = np.sum(ig_attr, axis=1)
        else:
            ig_attr=None

        risk = float(-torch.sum(survival, dim=1).detach().cpu().numpy())
        event_time = float(event_time)
        c = float(c)
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(),
                                           'survival': event_time, 'censorship': c,'ig': ig_attr}})

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index

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
        results_pkl_path = os.path.join(args.results_dir, 'split_latest_{}_results.pkl'.format(i))

        ### Gets the Train + Val Dataset Loader.
        train_dataset, val_dataset = dataset.return_splits(from_id=False,
                                                           csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        # --- split sanity check (unit=slide_id): train/val must not share slides ---
        train_slides = set(train_dataset.slide_data["slide_id"].astype(str).tolist())
        val_slides = set(val_dataset.slide_data["slide_id"].astype(str).tolist())
        overlap = train_slides & val_slides
        assert len(overlap) == 0, f"[SPLIT ERROR] train/val share slide_id: {list(sorted(overlap))[:10]}"

        # --- patient overlap check (if patient_id is available) ---
        if "patient_id" in train_dataset.slide_data.columns and "patient_id" in val_dataset.slide_data.columns:
            train_patients = set(train_dataset.slide_data["patient_id"].astype(str).tolist())
            val_patients = set(val_dataset.slide_data["patient_id"].astype(str).tolist())
            p_overlap = train_patients & val_patients
            assert len(p_overlap) == 0, f"[SPLIT ERROR] train/val share patient_id: {list(sorted(p_overlap))[:10]}"

        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)

        ### Run Train-Val on Survival Task.
        if args.task_type == 'survival':
            train_split, val_split = datasets

            print('\nInit Model...', end=' ')
            dropout = 0.25 if args.drop_out else 0.0
            model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
            args.fusion = None if args.fusion == 'None' else args.fusion


            if args.model_type == 'mcat':
                model_dict = {
                    'fusion': args.fusion,
                    'n_classes': args.n_classes,
                    'dropout': dropout,
                    'transformer_mode': args.transformer_mode,
                    'pooling': args.pooling,
                }
                model = MCAT_Surv(**model_dict).to(device)
            else:
                raise NotImplementedError

            print('Done!')
            print_network(model)


            print('\nInit Loaders...', end=' ')
            train_loader = get_split_loader(train_split, mode=args.mode, batch_size=args.batch_size)
            val_loader = get_split_loader(val_split, mode=args.mode, batch_size=args.batch_size)
            print('Done!')

            model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(i)), map_location=device))
            use_ig=False
            results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes,use_ig)
            print('Val c-Index: {:.4f}'.format(val_cindex))
            latest_val_cindex.append(val_cindex)

        ### Write Results for Each Split to PKL
        save_pkl(results_pkl_path, {'val': results_val_dict})
        end_time = timer()
        print('Fold %d Time: %f seconds' % (i, end_time - start_time))

        ### Finish 5-Fold CV Evaluation.
        final_df1 = pd.DataFrame(latest_val_cindex)
        final_df1['folds'] = final_df1.index
        final_df1['folds'] = final_df1['folds'] + start
        final_df = pd.DataFrame({'folds': np.arange(start, i + 1)})
        final_df = pd.merge(final_df, final_df1, how='left', on=['folds'])
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
parser.add_argument('--log_data', action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Whether or not to overwrite experiments (if already ran)')
parser.add_argument('--project_name', type=str, default='tcga-stad', help='prject name')
parser.add_argument('--task', type=str, default='survival',
                    help="Task name; should contain 'survival' for this script (Default: survival)")


### Model Parameters.
parser.add_argument('--model_type', type=str, choices=['snn', 'deepset', 'amil', 'mi_fcn', 'mcat'], default='mcat',
                    help='Type of model (Default: mcat)')
parser.add_argument('--mode', type=str, choices=['coattn'], default='coattn',
                    help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion', type=str, choices=['None', 'concat', 'bilinear'], default='concat',
                    help='Type of fusion. (Default: concat).')
parser.add_argument('--transformer_mode', type=str, choices=['separate', 'shared'], default='separate',
                    help="Transformer weights across modalities.")
parser.add_argument('--pooling', type=str, choices=['gap', 'attn'], default='attn',
                    help="Global pooling after transformer.")
parser.add_argument('--apply_sig', action='store_true', default=False,
                    help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats', action='store_true', default=False,
                    help='Use genomic features as tabular features.')
parser.add_argument('--drop_out', action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi', type=str, default='small', help='Network size of AMIL model')

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
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='Enable weighted sampling')
parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.split_dir = args.project_name
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
args = get_custom_exp_code(args)

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

encoding_size = 1024
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
    raise NotImplementedError

if isinstance(dataset, Generic_MIL_Survival_Dataset):
    args.task_type = 'survival'
else:
    raise NotImplementedError

### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

### Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code,
                                str(args.exp_code))
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
