# HEX
===========
## AI-enabled virtual spatial proteomics from histopathology for interpretable biomarker discovery in lung cancer

**Abstract:** Spatial proteomics enables high-resolution mapping of protein expression and can transform our understanding of biology and disease. However, major challenges remain for clinical translation, including cost, complexity and scalability. Here we present H&E to protein expression (HEX), an AI model designed to computationally generate spatial proteomics profiles from standard histopathology slides. Trained and validated on 819,000 histopathology image tiles with matched protein expression from 382 tumor samples, HEX accurately predicts the expression of 40 biomarkers encompassing immune, structural and functional programs. HEX demonstrates substantial performance gains over alternative methods for protein expression prediction from H&E images. We develop a multimodal data integration approach that combines the original H&E image and AI-derived virtual spatial proteomics to enhance outcome prediction. Applied to six independent non-small-cell lung cancer cohorts totaling 2,298 patients, HEX-enabled multimodal integration improved prognostic accuracy by 22% and immunotherapy response prediction by 24–39% compared with conventional clinicopathological and molecular biomarkers. Biological interpretation revealed spatially organized tumor–immune niches predictive of therapeutic response, including the co-localization of T helper cells and cytotoxic T cells in responders, and immunosuppressive tumor-associated macrophage and neutrophil aggregates in non-responders. HEX provides a low-cost and scalable approach to study spatial biology and enables the discovery and clinical translation of interpretable biomarkers for precision medicine.

## Dependencies:

**Hardware:**
- NVIDIA GPU (Tested on NVIDIA L40S x8) with CUDA 11.8 and cuDNN 9.1 (Ubuntu 22.04)

**Software:**
- Python (3.10.15), PyTorch (2.4.0+cu118)

**Additional Python Libraries:**
- accelerate (1.2.0), captum (0.7.0),fsspec (2024.10.0), ftfy (6.3.1), gitpython (3.1.43), h5py (3.12.1), huggingface-hub (0.26.5), imageio (2.36.1), joblib (1.4.2), lifelines (0.30.0), lightning-utilities (0.11.9), lxml (5.3.0), matplotlib (3.9.3), musk (1.0.0), networkx (3.4.2), nltk (3.9.1), numpy (2.2.0), opencv-python (4.10.0.84), openslide-python (1.4.1), pandas (2.2.3), pillow (11.0.0), protobuf (5.29.1), pytorch-lightning (2.2.1), scikit-image (0.24.0), scikit-learn (1.5.2), scikit-survival (0.23.1), scipy (1.14.1), seaborn (0.13.2), tensorboardx (2.6.2.2), timm (0.9.8), torch-geometric (2.6.1), torchaudio (2.5.1), torchvision (0.20.1), tqdm (4.67.1), transformers (4.47.0), wandb (0.19.1)
* MUSK (https://github.com/lilab-stanford/MUSK)
* Palom (https://github.com/labsyspharm/palom)
* DINOv2 (https://github.com/facebookresearch/dinov2)
* CLAM (https://github.com/mahmoodlab/CLAM)
* imbalanced-regression (https://github.com/YyzHarry/imbalanced-regression)
* robust_loss_pytorch (https://github.com/jonbarron/robust_loss_pytorch)
* MCAT (https://github.com/mahmoodlab/MCAT)


## Step 1: Preprocessing CODEX and H&E images
* Use the palom package to co-register CODEX and H&E images and obtain the registered CODEX images.
* Run `extract_marker_info_patch.py` to extract protein expression for each image patch.
* Construct the dataset with paired histopathology images and matched protein expression using the `extract_he_patch.py` script.
* Create patient-level splits for both HEX and MICA using CLAM-style split utilities (from `mahmoodlab/CLAM`) based on your cohort metadata, then run `check_splits.py` to sanity-check split integrity.

## Step 2: train and test HEX
* Start training using `torchrun --nnodes=1 --nproc-per-node=8 ./hex/train_dist_codex_lung_marker.py`. 
Logs and checkpoints will be saved to writer_dir and checkpoint_dir, respectively.
* Evaluate your model checkpoint by running `python test_codex_lung_marker.py` with checkpoint_path specify the `<save_location>/models/your_checkpoint.pth`. 
Output results will be stored in `save_dir`. To get you started, example data are provided in the folder `hex/sample_data`.

## Step 3: train and test MICA
* Use CLAM to preprocess WSIs and generate histology feature bags (MCAT-style pipeline).
* Apply a trained HEX model to generate the corresponding CODEX images for each WSI, then run `codex_h5_png2fea.py` to construct CODEX feature bags (DINOv2).
* Train MICA with `train_mica.py`, e.g. `python train_mica.py --mode coattn --base_path your_path --gc 8 --project_name your_project --max_epochs 20 --lr 1e-5`. Training logs and checkpoints will be saved under `results_dir`.
  Example data are provided in `mica/sample_data`. Your data layout should follow the MCAT format.
* Evaluate a checkpoint with `test_mica.py` (results will be saved as a `.pkl` in the corresponding results folder).
* For interpretability, `test_mica.py` can also compute integrated gradients to visualize spatial patterns.


## Acknowledgments
This project builds upon many open-source repositories such as CLAM (https://github.com/mahmoodlab/CLAM), MCAT (https://github.com/mahmoodlab/MCAT), imbalanced-regression (https://github.com/YyzHarry/imbalanced-regression), and Palom (https://github.com/labsyspharm/palom). We thank the authors and contributors to these repositories.
## License
This repository is licensed under the CC-BY-NC-ND 4.0 license. This repository includes/depends on third-party components that are licensed under their respective licenses . Please refer to each project for details.
## Citation
If you find our work useful in your research, please consider citing:
* Li, Z., Li, Y., Xiang, J. et al. AI-enabled virtual spatial proteomics from histopathology for interpretable biomarker discovery in lung cancer. Nat Med 32, 231–244 (2026). https://doi.org/10.1038/s41591-025-04060-4



