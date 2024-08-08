from typing import Dict
import torch
import torch.nn as nn
from UniT.model.tactile.utils import quaternion_angle_loss
from UniT.model.tactile.utils import ResNetPerception

class ResnetPerception(nn.Module):
    def __init__(self, 
            output_dim = 4,
            backbone = 'resnet34',
            pretrained = False,
        ):
        super().__init__()
        self.cnn = ResNetPerception(num_classes=output_dim, backbone=backbone, pretrained=pretrained)


    def forward(self, image):
        # image shape B H W C
        # convert to B C H W
        image = image.permute(0, 3, 1, 2)
        output = self.cnn(image)
        return output


    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        y_hat = self.forward(batch['image'])
        y = batch['3Dpose']
        loss = quaternion_angle_loss(y_hat, y)
        return loss