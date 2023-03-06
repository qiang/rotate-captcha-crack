import torch.nn as nn
from torch import Tensor
from torchvision import models

from ..const import ROTNET_CLS_NUM


class RotNet(nn.Module):
    """
    Args:
        train (bool, optional): True to load the pretrained parameters. Defaults to True.

    Note:
        impl: `rotnet_street_view_resnet50` in https://github.com/d4nst/RotNet
    """

    def __init__(self, train: bool = True) -> None:
        super(RotNet, self).__init__()

        weights = models.ResNet50_Weights.DEFAULT if train else None
        self.backbone = models.resnet50(weights=weights)

        fc_channels = self.backbone.fc.in_features
        del self.backbone.fc
        self.backbone.fc = nn.Linear(fc_channels, ROTNET_CLS_NUM)

        if train:
            nn.init.xavier_normal_(self.backbone.fc.weight)
            nn.init.zeros_(self.backbone.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward

        Args:
            x (Tensor): img_tensor ([N,C,H,W]=[batch_size,3,224,224], dtype=float32, range=[0.0,1.0))

        Returns:
            Tensor: predict result ([N,C]=[batch_size,ROTNET_CLS_NUM), dtype=float32, range=[0.0,1.0))
        """

        x = self.backbone.forward(x)

        return x

    def predict(self, img_ts: Tensor) -> float:
        """
        predict the counter clockwise rotation angle

        Args:
            img_ts (Tensor): img_tensor ([C,H,W]=[3,224,224], dtype=float32, range=[0.0,1.0))

        Returns:
            float: predict result. range=[0.0,1.0)

        Note:
            Use Image.rotate(-ret * 360) to recover the image.
        """

        img_ts = img_ts.unsqueeze_(0)

        onehot_ts = self.backbone.forward(img_ts)
        angle = float(onehot_ts.argmax(1).cpu().item()) / ROTNET_CLS_NUM

        return angle
