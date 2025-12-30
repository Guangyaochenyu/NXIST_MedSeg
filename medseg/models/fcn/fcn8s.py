from collections import OrderedDict

from torch import nn, Tensor

from . import FCN, FCNBackbone, FCNClassifier

__all__ = [ 'FCN8s' ]

class FCN8sBackbone(FCNBackbone):
    def __init__(self) -> None:
        super(FCN8sBackbone, self).__init__()

    def forward(self, x: Tensor) -> dict[str, dict[str, Tensor]]:
        result          = OrderedDict()
        out             = OrderedDict()
        h               = x
        h               = self.conv1(h)
        h               = self.conv2(h)
        h               = self.conv3(h)
        pool3           = h
        h               = self.conv4(h)
        pool4           = h
        h               = self.conv5(h)
        h               = self.fc6(h)
        h               = self.fc7(h)
        out['pool3']    = pool3
        out['pool4']    = pool4
        out['h']        = h
        out['x_size']   = x.size()
        result['out']   = out
        return result

class FCN8sClassifier(FCNClassifier):
    def __init__(
            self,
            num_classes: int = 21
        ) -> None:
        super(FCNClassifier, self).__init__()
        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, bias=False)
        self._initialize_weights()
    def forward(self, x: dict[str, Tensor]) -> Tensor:
        h               = x['h']
        pool3           = x['pool3']
        pool4           = x['pool4']
        x_size          = x['x_size']

        h               = self.score_fr(h)
        h               = self.upscore2(h)
        upscore2        = h
        h               = self.score_pool4(pool4)
        h               = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c    = h
        h               = upscore2 + score_pool4c
        h               = self.upscore_pool4(h)
        upscore_pool4   = h
        h               = self.score_pool3(pool3)
        h               = h[:, :,
                                9:9 + upscore_pool4.size()[2],
                                9:9 + upscore_pool4.size()[3]]
        score_pool3c    = h
        h               = upscore_pool4 + score_pool3c
        h               = self.upscore8(h)
        h               = h[:, :, 31:31 + x_size[2], 31:31 + x_size[3]].contiguous()
        return h

class FCN8s(FCN):
    def __init__(
            self,
            num_classes: int = 21
        ) -> None:
        super(FCN8s, self).__init__(
            backbone       = FCN8sBackbone(),
            classifier     = FCN8sClassifier(num_classes),
            aux_classifier = None
        )