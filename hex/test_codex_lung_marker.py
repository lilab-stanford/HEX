import os
from os.path import join
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.stats import pearsonr
from tqdm import tqdm
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torch.cuda.amp import autocast
from hex_architecture import CustomModel
from utils import PatchDataset

# Define biomarker names
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

def load_model(checkpoint_path):
    try:
        model = CustomModel(visual_output_dim=1024, num_outputs=40)
        sd = torch.load(checkpoint_path, map_location='cpu')
        incompat = model.load_state_dict(sd, strict=False)
        print(f"[load_state_dict] missing_keys={len(incompat.missing_keys)} unexpected_keys={len(incompat.unexpected_keys)}")
        model = nn.DataParallel(model).cuda()
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def main():
    device='cuda'
    save_dir = ""
    # Load the model
    checkpoint_path = join(save_dir, 'checkpoint.pth')# replace with your checkpoint path or the provided demo checkpoint (pipeline-only, not the paper-trained checkpoints)
    model = load_model(checkpoint_path)

    # Prepare the dataset
    data_dir = "./sample_data/"
    img_dir = Path(data_dir) / "he_patches"
    csv_dir = Path(data_dir) / "channel_registered"

    print("Loading data...")
    # Filter patients if needed (default: use sample_data/splits_0.csv if present)
    all_ids = [csv_file.stem for csv_file in csv_dir.glob('*.csv')]
    patient_ids = all_ids

    split_csv = Path(data_dir) / "splits_0.csv"
    if split_csv.exists():
        split_df = pd.read_csv(split_csv)
        patient_ids = [str(int(x)) for x in split_df["val"].dropna().tolist()]

    all_patches = []
    for val_id in tqdm(patient_ids, desc="Loading CSVs"):
        val_csv = pd.read_csv(join(csv_dir, f'{val_id}.csv'))
        val_csv['patch_id'] = val_csv['slide'].astype(str) + '_' + val_csv['index'].astype(str)
        all_patches.append(val_csv)
    all_patches = pd.concat(all_patches)
    all_patches.reset_index(drop=True, inplace=True)
    all_patches['images'] = str(img_dir) + '/' + all_patches['slide'].astype(str) + '/' + all_patches['slide'].astype(
        str) + '_' + all_patches['index'].astype(str) + '.png'
    all_patches.reset_index(drop=True, inplace=True)
    print(f"Total patches: {len(all_patches)}")

    label_columns = [f'mean_intensity_channel{i}' for i in range(1, 41)]
    transform = transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])
    dataset = PatchDataset(all_patches, label_columns, transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True)

    # Perform inference
    all_preds = []
    all_labels = []
    all_image_paths = []
    print("Starting inference...")

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        val_loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in val_loop:
            inputs, labels = data[0].to(device,non_blocking=True), data[1].to(torch.float32)
            image_paths = data[2]
            outputs,_ = model(inputs)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.numpy())
            all_image_paths.extend(image_paths)


    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    torch.cuda.empty_cache()

    patch_results = []
    for i, image_path in enumerate(all_image_paths):
        slide_id, patch_index = Path(image_path).stem.split('_')
        patch_data = {
            'slide_id': slide_id,
            'patch_index': patch_index
        }
        for j, biomarker in biomarker_names.items():
            patch_data[f'{biomarker}_pred'] = all_preds[i, j-1]
            patch_data[f'{biomarker}_label'] = all_labels[i, j-1]
        patch_results.append(patch_data)

    patch_df = pd.DataFrame(patch_results)
    patch_df.to_csv(join(save_dir,"patch_predictions.csv"), index=False)
    print("Patch-level predictions and labels saved to patch_predictions.csv")

    print("Calculating Pearson correlations...")
    pearson_r_values = []
    for i in tqdm(range(all_labels.shape[1]), desc="Calculating correlations"):
        r, _ = pearsonr(all_labels[:, i].astype(np.float64), all_preds[:, i].astype(np.float64))
        pearson_r_values.append(r)

    results = []
    for i, r in enumerate(pearson_r_values):
        biomarker = biomarker_names[i+1]
        print(f"{biomarker}: Pearson R = {r:.4f}")
        results.append({"Biomarker": biomarker, "Pearson_R": r})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Pearson_R", ascending=False)
    results_df.to_csv(join(save_dir,"biomarker_pearson_r.csv"), index=False)
    print("Results saved to biomarker_pearson_r.csv")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average Pearson R: {np.nanmean(pearson_r_values):.4f}")
    print(f"Median Pearson R: {np.nanmedian(pearson_r_values):.4f}")
    print(f"Min Pearson R: {np.nanmin(pearson_r_values):.4f}")
    print(f"Max Pearson R: {np.nanmax(pearson_r_values):.4f}")

    # Print top and bottom 5 biomarkers
    print("\nTop 5 Biomarkers:")
    print(results_df.head().to_string(index=False))
    print("\nBottom 5 Biomarkers:")
    print(results_df.tail().to_string(index=False))

if __name__ == "__main__":
    main()
