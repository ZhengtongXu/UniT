from typing import Dict
import torch
import torch.nn as nn
from UniT.model.tactile.utils import quaternion_angle_loss
import hydra
import numpy as np
from t3.models import T3
from t3.models import encoder
from t3.models.decoder import CNNFCDecoder
from omegaconf import OmegaConf
from UniT.model.tactile.utils import MlpHead

class T3Perception(nn.Module):
    def __init__(self, 
            encoder_path,
            trunk_path,
            mini: nn.Module,
            shared_trunk: nn.Module,
            encoder_embed_dim,
            device = "cuda:0" ,
            freeze_encoder=True,
        ):
        super().__init__()

        self.encoder = mini.to(device)
        self.encoder.load(encoder_path)
        self.trunk = shared_trunk.to(device)
        self.trunk.load(trunk_path)

        if freeze_encoder:
            self.encoder.freeze()
            self.trunk.freeze()

        self.head = MlpHead(encoder_embed_dim).to(device)

        # to device
        self.encoder = self.encoder.to(device)
        self.trunk = self.trunk.to(device)
        self.head = self.head.to(device)

    def forward(self, image):
        # batch: B H W C
        image = image.permute(0, 3, 1, 2)
        tactile_feature = self.encoder(image)
        tactile_feature = self.trunk(tactile_feature)
        output = self.head(tactile_feature[:,0])
        return output

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        image = batch['image']
        y_hat = self.forward(image)
        y = batch['3Dpose']
        loss = quaternion_angle_loss(y_hat, y)
        return loss