import torch.nn as nn
import torch


class VGGNet(nn.Module):
    def __init__(self, cfg, n_classes=1000):
        super(VGGNet, self).__init__()
        self.layers = self._make_conv_layers(cfg)
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, n_classes)
        )


    def _make_conv_layers(self, cfg, in_channels=3):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(2)]
            elif x == 'L':
                layers += [nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)]
            else:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1), nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




cfgs = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'A-LRN': [64, 'L', 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }

def vgg11(n_classes=1000):
    return VGGNet(cfgs['A-LRN'], n_classes)


def vgg13(n_classes=1000):
    return VGGNet(cfgs['B'], n_classes)


def vgg16(n_classes=1000):
    return VGGNet(cfgs['D'], n_classes)


def vgg19(n_classes=1000):
    return VGGNet(cfgs['E'], n_classes)


if __name__ == '__main__':

    for cfg in cfgs.keys():
        net = VGGNet(cfg=cfgs[cfg])
        X = torch.randn(1, 3, 224, 224)
        out = net(X)
        print(out.shape)
