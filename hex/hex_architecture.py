
import torch
import torch.nn as nn
from PIL import Image
from timm import create_model
from musk import utils


class CustomModel(nn.Module):
    def __init__(self, visual_output_dim, num_outputs):
        super(CustomModel, self).__init__()
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

    def forward(self, x):
        x = self.visual(
            image=x,
            with_head=False,
            out_norm=False
        )[0]
        features = self.regression_head(x)
        preds = self.regression_head1(features)
        return preds, features
