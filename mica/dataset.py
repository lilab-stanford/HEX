from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv', mode='omic', apply_sig=False,
                seed=7, print_info=True, n_bins=4, ignore=[],
                 patient_strat=False, label_col=None, filter_dict={}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset

        Args:
            csv_file (string): Path to the csv file with annotations.
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None

        slide_data = pd.read_csv(csv_path, low_memory=False)

        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        slide_data['case_id'] = slide_data['case_id'].astype(str)
        slide_data['slide_id'] = slide_data['slide_id'].astype(str)

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps

        disc_labels, q_bins = pd.cut(slide_data[label_col], bins=q_bins, retbins=True, labels=False, right=False,
                                     include_lowest=True)
        slide_data.insert(2, 'label', disc_labels.values.astype(int))

        slide_data.reset_index(drop=True, inplace=True)

        # - Training/evaluation unit is **slide** (one slide = one sample).
        # - The split CSV is assumed to guarantee:
        #     (i) train column contains all slides from training patients
        #     (ii) val column contains exactly one slide per patient
        slide_data = slide_data.rename(columns={'case_id': 'patient_id'})
        slide_data = slide_data.assign(case_id=slide_data['slide_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins) - 1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c): key_count})
                key_count += 1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes = len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id': patients_df['case_id'].values, 'label': patients_df['label'].values}

        new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12]
        self.mode = mode
        self.cls_ids_prep()

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id']))  # get unique patients
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]]  # get patient label
            patient_labels.append(label)

        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            print('Unit-LVL (slide); Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    def get_split_from_df(self, all_splits: dict, split_key: str = 'train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode,
                                  data_dir=self.data_dir, codex_path=self.codex_deep, label_col=self.label_col,
                                  num_classes=self.num_classes)
        else:
            split = None

        return split

    def return_splits(self, from_id: bool = True, csv_path: str = None, all_train: bool = False):
        if from_id:
            raise NotImplementedError
        else:
            if not all_train:
                assert csv_path
                all_splits = pd.read_csv(csv_path)
                train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
                val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
                test_split = None

                return train_split, val_split
            else:
                train_split = Generic_Split(self.slide_data, metadata=self.metadata, mode=self.mode,
                                            data_dir=self.data_dir,
                                            codex_path=self.codex_deep, label_col=self.label_col,
                                            num_classes=self.num_classes)
                return train_split, None

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, codex_deep, mode: str = 'omic', **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.codex_deep = codex_deep

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = [case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if self.mode != 'coattn':
            raise NotImplementedError("This repo only supports mode='coattn' (path+codex).")

        path_features = []
        codex_features = []
        for slide_id in slide_ids:
            wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(os.path.splitext(slide_id)[0]))
            wsi_bag = torch.load(wsi_path)
            path_features.append(wsi_bag)
            codex_deep = self.codex[os.path.splitext(slide_id)[0]]
            codex_deep = torch.tensor(codex_deep, dtype=torch.float32)
            codex_features.append(codex_deep)
        path_features = torch.cat(path_features, dim=0)
        codex_features = torch.cat(codex_features, dim=0)
        return (path_features, codex_features, label, event_time, c)


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, mode, data_dir=None, codex_path=None, label_col=None,
                 num_classes=2):
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        features_dict = {}
        with h5py.File(codex_path, 'r') as h5_file:
            for name in h5_file.keys():
                features_dict[name] = h5_file[name][:]
        self.codex = features_dict
        self.num_classes = num_classes
        self.label_col = label_col
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)
