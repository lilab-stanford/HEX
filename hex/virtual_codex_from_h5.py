import os
from os.path import join
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py
import openslide


def check_mag(wsi):
    # Best-effort magnification inference from MPP; default to 40x if unknown.
    try:
        currMPP = float(wsi.properties["aperio.MPP"])
    except Exception:
        try:
            currMPP = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])
        except Exception:
            currMPP = 0.25

    if currMPP < 0.2:
        mag = 80
    elif 0.2 <= currMPP < 0.3:
        mag = 40
    elif 0.4 <= currMPP < 0.6:
        mag = 20
    else:
        mag = 40
    return mag

# TODO: fill these paths before running
slide_ext = ".svs"
he_dir = Path("")         # TODO: directory containing WSIs
codex_h5_dir = Path("")   # TODO: directory containing per-slide *.h5 predictions
save_dir = Path("")       # TODO: output directory
os.makedirs(save_dir, exist_ok=True)

codex_h5_files = sorted(codex_h5_dir.glob("*.h5"))
for h5_file in tqdm(codex_h5_files):
    he_id = h5_file.stem
    he_path = he_dir / f"{he_id}{slide_ext}"
    if not he_path.exists():
        print(f"[skip] missing WSI: {he_path}")
        continue

    wsi = openslide.open_slide(str(he_path))
    mag = check_mag(wsi)

    scale_down_factor = int(224 / (40 / mag))
    width = wsi.dimensions[0] // scale_down_factor + 1
    height = wsi.dimensions[1] // scale_down_factor + 1
    wsi.close()

    with h5py.File(str(h5_file), "r") as f:
        codex_prediction = f["codex_prediction"][:]  # (N, C)
        coords = f["coords"][:]                      # (N, 2)

    C = codex_prediction.shape[1]
    codex_image = np.zeros((height, width, C), dtype=np.float16)

    for i in range(len(coords)):
        x, y = coords[i]
        x, y = int(x / scale_down_factor), int(y / scale_down_factor)
        if 0 <= x < width and 0 <= y < height:
            codex_image[y, x] = codex_prediction[i]

    save_path = join(str(save_dir), f"{he_id}.npy")
    np.save(save_path, codex_image)
