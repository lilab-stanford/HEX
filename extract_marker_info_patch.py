import os
import palom
import pandas as pd
import numpy as np
from pathlib import Path
from os.path import join
from tqdm import tqdm
import h5py
import multiprocessing
from matplotlib import pyplot as plt

# Define directories and file paths
he_dir = ''
codex_dir = ''
he_list = list(Path(he_dir).glob('*.svs'))

he_codex_map = pd.read_csv('.he_codex_map.csv')
he_codex_map['codex'] = he_codex_map['codex'].str.replace('.ome', '')
h5_dir = ''

for he_path in he_list:
    he_id = os.path.basename(he_path).split('.')[0]
    codex_id_row = he_codex_map[he_codex_map['he'] == int(he_id)]
    codex_id = codex_id_row['codex'].values[0]
    dapi_path = join(codex_dir, codex_id + '.ome.tiff')

    c2r = palom.reader.OmePyramidReader(dapi_path)
    h5_file_path = join(h5_dir, f'{he_id}.h5')
    with h5py.File(h5_file_path, 'r') as hdf5_file:
        coords = hdf5_file['coords'][:]
        patch_level = hdf5_file['coords'].attrs['patch_level']
        patch_size = hdf5_file['coords'].attrs['patch_size']
        length = len(coords)

    # Prepare DataFrame to store results
    patches_df = pd.DataFrame({
        'slide': he_id,
        'index': np.arange(length),
        'x': coords[:, 0],
        'y': coords[:, 1]
    })

    def process_patch(args):
        idx, x, y = args
        slc = (
            slice(None),  # All channels
            slice(y, y + patch_size),
            slice(x, x + patch_size)
        )
        patch = c2r.pyramid[0][slc].compute()
        mean_intensity = patch.reshape(patch.shape[0], -1).mean(axis=1)
        return idx, mean_intensity

    patch_args = [(idx, x, y) for idx, (x, y) in enumerate(coords)]

    num_workers = multiprocessing.cpu_count()  # Adjust if needed
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use tqdm to show progress bar
        results = list(tqdm(pool.imap_unordered(process_patch, patch_args), total=length, desc=f'Processing patches for slide {he_id}'))

    results.sort(key=lambda x: x[0])
    # Extract mean intensities
    mean_intensities_array = np.array([res[1] for res in results])

    # Add mean intensities to DataFrame
    num_channels = mean_intensities_array.shape[1]
    channel_columns = [f'mean_intensity_channel{ch+1}' for ch in range(num_channels)]
    mean_intensities_df = pd.DataFrame(mean_intensities_array, columns=channel_columns)
    patches_df = pd.concat([patches_df.reset_index(drop=True), mean_intensities_df.reset_index(drop=True)], axis=1)

    # Save the DataFrame to CSV
    output_csv_path = f'./{he_id}.csv'
    patches_df.to_csv(output_csv_path, index=False)
