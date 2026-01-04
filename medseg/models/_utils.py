from collections import OrderedDict
from typing import Optional

import os
import torch
from torch import nn, Tensor
from torch.nn import functional as F

class BasicModule(nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        return self

    def save(self, type='last'):
        prefix = os.path.join(
            os.environ.get('MEDSEG_HOME'), 
            'checkpoints', 
            f"{os.environ.get('MEDSEG_MODEL')}-{os.environ.get('MEDSEG_DATA')}",
            os.environ.get('MEDSEG_TIME'),
        )
        os.makedirs(prefix, exist_ok=True)
        name = os.path.join(prefix, f"{type}.pth")
        torch.save(self.state_dict(), name)
        return name


class _SimpleSegmentationModel(BasicModule):
    __constants__ = ["aux_classifier"]

    def __init__(
            self,
            backbone:       nn.Module, 
            classifier:     nn.Module, 
            aux_classifier: Optional[nn.Module] = None
        ) -> None:
        super(_SimpleSegmentationModel, self).__init__()

        self.backbone       = backbone
        self.classifier     = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        input_shape = x.shape[-2:]
        features    = self.backbone(x)

        result      = OrderedDict()
        x           = features["out"]
        x           = self.classifier(x)
        x           = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x       = features["aux"]
            x       = self.aux_classifier(x)
            x       = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result