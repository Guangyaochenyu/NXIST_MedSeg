from collections import OrderedDict

from torch import nn, Tensor

from . import FCN, FCNBackbone, FCNClassifier

__all__ = [ 'FCN32s' ]

class FCN32sBackbone(FCNBackbone):
    def __init__(self) -> None:
        super(FCN32sBackbone, self).__init__()

    def forward(self, x: Tensor) -> dict[str, dict[str, Tensor]]:
        result          = OrderedDict()
        out             = OrderedDict()
        h               = x
        h               = self.conv1(h)
        h               = self.conv2(h)
        h               = self.conv3(h)
        h               = self.conv4(h)
        h               = self.conv5(h)
        h               = self.fc6(h)
        h               = self.fc7(h)
        out['h']        = h
        out['x_size']   = x.size()
        result['out']   = out
        return result

class FCN32sClassifier(FCNClassifier):
    def __init__(
            self,
            num_classes: int = 21
        ) -> None:
        super(FCN32sClassifier, self).__init__()
        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32,
                                          bias=False)
        self._initialize_weights()
    def forward(self, x: dict[str, Tensor]) -> Tensor:
        h               = x['h']
        x_size          = x['x_size'] 

        h               = self.score_fr(h)
        h               = self.upscore(h)
        h               = h[:, :, 19:19 + x_size[2], 19:19 + x_size[3]].contiguous()
        return h

class FCN32s(FCN):
    def __init__(
            self,
            num_classes: int = 21
        ) -> None:
        super(FCN32s, self).__init__(
            backbone       = FCN32sBackbone(),
            classifier     = FCN32sClassifier(num_classes),
            aux_classifier = None
        )