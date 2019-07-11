import torch
import torch.nn as nn
import torch.nn.functional as F

from ActiveShift2d import ActiveShift2d

# BN-ReLU-1x1Conv-BN-ReLU-ASL-1x1Conv
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.asl = ActiveShift2d(planes, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)

        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.asl(out)
        out = self.conv2(out)
        out += shortcut
        return out

class AS_ResNet(nn.Module):
    def __init__(self, block, repeat, num_classes=10):
        super(AS_ResNet, self).__init__()
        self.base_width = 16
        w = self.base_width

        self.conv1 = nn.Conv2d(3, self.base_width, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer1 = self._make_layer(block, w, w, repeat[0], stride=1)
        self.layer2 = self._make_layer(block, w, w, repeat[1], stride=2)
        self.layer3 = self._make_layer(block, w, 2 * w, repeat[2], stride=2)
        self.layer4 = self._make_layer(block, 2 * w, 4 * w, repeat[3], stride=2)
        self.layer5 = self._make_layer(block, 4 * w, 8 * w, repeat[4], stride=2)

        self.fc_layer = nn.Linear(8 * w, num_classes)

    def _make_layer(self, block, inplanes, planes, repeat, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride))
        for _ in range(1, repeat):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out

if __name__ == '__main__':
    model = AS_ResNet(BasicBlock, [1,3,4,6,3], num_classes=10)
    x = torch.Tensor(1,3,224,224)
    y = model(x)
    print(y.size())