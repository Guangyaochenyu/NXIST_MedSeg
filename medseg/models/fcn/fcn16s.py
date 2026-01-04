from collections import OrderedDict

from torch import nn, Tensor

from . import FCN, FCNBackbone, FCNClassifier

__all__ = [ 'FCN16s' ]

class FCN16sBackbone(FCNBackbone):
    def __init__(self) -> None:
        super(FCN16sBackbone, self).__init__()

    def forward(self, x: Tensor) -> dict[str, dict[str, Tensor]]:
        result          = OrderedDict()
        out             = OrderedDict()
        h               = x
        h               = self.conv1(h)
        h               = self.conv2(h)
        h               = self.conv3(h)
        h               = self.conv4(h)
        pool4           = h
        h               = self.conv5(h)
        h               = self.fc6(h)
        h               = self.fc7(h)
        out['pool4']    = pool4
        out['h']        = h
        out['x_size']   = x.size()
        result['out']   = out
        return result

class FCN16sClassifier(FCNClassifier):
    def __init__(
            self,
            num_classes: int = 21
        ) -> None:
        super(FCNClassifier, self).__init__()
        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(
            num_classes, num_classes, 32, stride=16, bias=False)
        self._initialize_weights()
    def forward(self, x: dict[str, Tensor]) -> Tensor:
        h               = x['h']
        pool4           = x['pool4']
        x_size          = x['x_size'] 

        h               = self.score_fr(h)
        h               = self.upscore2(h)
        upscore2        = h
        h               = self.score_pool4(pool4)
        h               = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c    = h
        h               = upscore2 + score_pool4c
        h               = self.upscore16(h)
        h               = h[:, :, 27:27 + x_size[2], 27:27 + x_size[3]].contiguous()
        return h    

class FCN16s(FCN):
    def __init__(
            self,
            num_classes: int = 21,
            **kwargs,
        ) -> None:
        super(FCN16s, self).__init__(
            backbone       = FCN16sBackbone(),
            classifier     = FCN16sClassifier(num_classes),
            aux_classifier = None
        )