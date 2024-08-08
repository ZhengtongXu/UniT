from typing import Dict
import torch
import torch.nn as nn
from UniT.model.mae.utils import *
from UniT.model.tactile.utils import quaternion_angle_loss
from UniT.model.tactile.utils import ConvPoolingHead
from UniT.model.tactile.utils import MlpHead

class MaePerception(nn.Module):
    def __init__(self, 
            mae_config,
            freeze_encoder=True,
        ):
        super().__init__()

        if mae_config.get('pt_path') is not None:
            self.mae = torch.load(mae_config['pt_path'], map_location=mae_config['device'])
            print(f"Loaded MAE from {mae_config['pt_path']}")
        else:
            self.mae = MAE_ViT(**mae_config)
        
        if freeze_encoder:
            for param in self.mae.parameters():
                param.requires_grad = False

        self.vit_classifier = ViT_Classifier(self.mae.encoder,num_classes=4)

    # vqvae to cnn to mlp
    def forward(self, image):
        # batch: B H W C
        image = image.permute(0, 3, 1, 2)
        output = self.vit_classifier(image)
        return output


    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        image = batch['image']
        y_hat = self.forward(image)
        y = batch['3Dpose']
        loss = quaternion_angle_loss(y_hat, y)
        return loss