from typing import Dict
import torch
import torch.nn as nn
from UniT.taming.models.vqgan import VQModel
from UniT.model.tactile.utils import quaternion_angle_loss
from UniT.model.tactile.utils import ConvPoolingHead
from UniT.model.tactile.utils import MlpHead

class VqvaePerception(nn.Module):
    def __init__(self, 
            vq_model_config,
            latent_shape=[3,28,28],
            freeze_encoder=True
        ):
        super().__init__()
        self.vqgan =  VQModel(**vq_model_config)

        if freeze_encoder:
            for param in self.vqgan.parameters():
                param.requires_grad = False

        self.conv_pooling_haed = ConvPoolingHead(latent_shape[0])
        self.mlp_head = MlpHead(
            self.conv_pooling_haed(torch.randn(1, latent_shape[0], latent_shape[1], latent_shape[2])).shape[1])


    def forward(self, image):
        # batch: B H W C
        tactile_feature = self.vqgan.to_latent(self.vqgan.get_input({'image': image},'image'))
        tactile_feature = self.conv_pooling_haed(tactile_feature)
        output = self.mlp_head(tactile_feature)
        return output


    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        image = batch['image']
        y_hat = self.forward(image)
        y = batch['3Dpose']
        loss = quaternion_angle_loss(y_hat, y)
        return loss