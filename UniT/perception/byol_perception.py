from typing import Dict
import torch
import torch.nn as nn
from UniT.model.tactile.utils import quaternion_angle_loss
from UniT.model.tactile.utils import MlpHead
from torchvision import models

class ByolPerception(nn.Module):
    def __init__(self, 
            byol_config,
            freeze_encoder=True,
        ):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)

        self.resnet.load_state_dict(torch.load(byol_config['pt_path']))
        print(f"Loaded BYOL pretrained from {byol_config['pt_path']}")
        if freeze_encoder:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # don't freeze the last layer
        self.resnet.fc = MlpHead(self.resnet.fc.in_features)
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        # to device
        self.resnet = self.resnet.to(byol_config['device'])

    def forward(self, image):
        # batch: B H W C
        image = image.permute(0, 3, 1, 2)
        output = self.resnet(image)
        return output


    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        image = batch['image']
        y_hat = self.forward(image)
        y = batch['3Dpose']
        loss = quaternion_angle_loss(y_hat, y)
        return loss