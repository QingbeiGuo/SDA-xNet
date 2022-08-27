import torch
import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual

        x = F.relu(out)

        return residual, out, x


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=8)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual

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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = self.bn2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet18'], model_root)
    return model


def resnet34(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet34'], model_root)
    return model


def resnet50(pretrained=False, model_root=None, **kwargs):
    #model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    model = ResNet(Bottleneck, [5, 6, 12, 5], **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet50'], model_root)
    return model


def resnet101(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet101'], model_root)
    return model


def resnet152(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet152'], model_root)
    return model
