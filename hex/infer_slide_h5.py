"""Run patch-level HEX inference at coordinates stored in a CLAM-style H5 file."""

import argparse
from contextlib import nullcontext

DEFAULT_PATCH_LEVEL = 0
DEFAULT_PATCH_SIZE = 224
NUM_MARKERS = 40


class SlidePatchDataset:
    """Read RGB patches from one WSI at level-0 CLAM coordinates."""

    def __init__(self, wsi_path, coords, patch_level, patch_size, transform):
        self.wsi_path = str(wsi_path)
        self.coords = coords
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.transform = transform
        self._slide = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        import openslide

        if self._slide is None:
            self._slide = openslide.open_slide(self.wsi_path)

        x, y = (int(value) for value in self.coords[index])
        patch = self._slide.read_region(
            (x, y), self.patch_level, (self.patch_size, self.patch_size)
        ).convert("RGB")
        return self.transform(patch)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply a HEX checkpoint to WSI patches at CLAM-style coordinates "
            "and write patch-level marker-expression predictions to H5."
        )
    )
    parser.add_argument("--wsi", required=True, help="Path to the input WSI.")
    parser.add_argument(
        "--coord-h5", required=True, help="CLAM-style H5 containing a coords dataset."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the HEX checkpoint.")
    parser.add_argument("--output-h5", required=True, help="Path for the output H5.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    import torch
    import torch.nn as nn
    from musk import modeling  # noqa: F401 - registers MUSK models with Timm

    from hex_architecture import CustomModel

    model = CustomModel(visual_output_dim=1024, num_outputs=NUM_MARKERS)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    incompat = model.load_state_dict(state_dict, strict=False)
    print(
        "[load_state_dict] "
        f"missing_keys={len(incompat.missing_keys)} "
        f"unexpected_keys={len(incompat.unexpected_keys)}"
    )
    if device.type == "cuda":
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    return model


def main(args):
    import h5py
    import numpy as np
    import torch
    from timm.data.constants import (
        IMAGENET_INCEPTION_MEAN,
        IMAGENET_INCEPTION_STD,
    )
    from torch.utils.data import DataLoader
    from torchvision import transforms

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    with h5py.File(args.coord_h5, "r") as input_h5:
        coords_dataset = input_h5["coords"]
        coords = coords_dataset[:]
        patch_level = int(
            coords_dataset.attrs.get("patch_level", DEFAULT_PATCH_LEVEL)
        )
        patch_size = int(coords_dataset.attrs.get("patch_size", DEFAULT_PATCH_SIZE))

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape [N, 2], got {coords.shape}")

    transform = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
            ),
        ]
    )
    dataset = SlidePatchDataset(
        args.wsi, coords, patch_level, patch_size, transform
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(args.checkpoint, device)

    predictions = []
    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast_context():
        for patches in dataloader:
            outputs, _ = model(patches.to(device, non_blocking=device.type == "cuda"))
            predictions.append(outputs.detach().cpu().numpy())

    if predictions:
        codex_prediction = np.concatenate(predictions, axis=0)
    else:
        codex_prediction = np.empty((0, NUM_MARKERS), dtype=np.float32)

    if codex_prediction.shape != (len(coords), NUM_MARKERS):
        raise RuntimeError(
            "Expected one 40-dimensional prediction per coordinate, got "
            f"{codex_prediction.shape} for {len(coords)} coordinates"
        )

    with h5py.File(args.output_h5, "w") as output_h5:
        output_coords = output_h5.create_dataset("coords", data=coords)
        output_coords.attrs["patch_level"] = patch_level
        output_coords.attrs["patch_size"] = patch_size
        output_h5.create_dataset("codex_prediction", data=codex_prediction)


if __name__ == "__main__":
    main(parse_args())
