import torch
import torch.nn as nn
import math
import torch.nn.functional as F


__all__ = ['SENet', 'se_resnet_18', 'se_resnet_34', 'se_resnet_50', 'se_resnet_101', 'se_resnet_152']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if planes == 64:
            self.globalAvgPool = nn.AvgPool2d(56, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out

        out += residual
        x = F.relu(out)

        return residual, out, x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if planes == 64:
            self.globalAvgPool = nn.AvgPool2d(56, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(in_features=planes * 4, out_features=round(planes / 4))
        self.fc2 = nn.Linear(in_features=round(planes / 4), out_features=planes * 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0),out.size(1),1,1)
        out = out * original_out

        out += residual
        x = F.relu(out)

        return residual, out, x


class BasicStage(nn.Module):
    def __init__(self, block, inplanes, planes, blocks, stride=1, r=16, L=64):
        super(BasicStage, self).__init__()

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        self.layers =  nn.Sequential(*layers)

        d = max(int(planes * block.expansion/r), L)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(planes * block.expansion, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(blocks):
            self.fcs.append(
                nn.Conv2d(d, planes * block.expansion, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, block in enumerate(self.layers):
            residual, out, x = block(x)  #[b,c,w,h]
            fea_Zi = out.clone()
            if i == 0:
              fea_Z = fea_Zi.unsqueeze_(dim=1)
            else:
              fea_Z = torch.cat([fea_Z, fea_Zi.unsqueeze_(dim=1)], dim=1)

        fea_F = torch.sum(fea_Z, dim=1)
        fea_u = self.gap(fea_F)
        fea_v = self.fc(fea_u)

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_v)
            if i == 0:
                vectors = vector.unsqueeze_(dim=1)
            else:
                vectors = torch.cat([vectors, vector.unsqueeze_(dim=1)], dim=1)
        vectors = self.softmax(vectors)
        fea_O = F.relu(torch.sum(fea_Z * vectors, dim=1))

        return fea_O


class SENet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(SENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = BasicStage(block, self.inplanes, 64, layers[0])
        self.inplanes = 256
        self.stage2 = BasicStage(block, self.inplanes, 128, layers[1], stride=2)
        self.inplanes = 512
        self.stage3 = BasicStage(block, self.inplanes, 256, layers[2], stride=2)
        self.inplanes = 1024
        self.stage4 = BasicStage(block, self.inplanes, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = self.bn2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def se_resnet_18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def se_resnet_34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet_50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #model = SENet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model = SENet(Bottleneck, [5, 6, 12, 5], **kwargs)
    return model


def se_resnet_101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def se_resnet_152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model