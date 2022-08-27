
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
__all__ = ['Res2Net', 'res2net50']


model_urls = {
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        x = F.relu(out)

        return residual, out, x


class BasicStage(nn.Module):
    def __init__(self, block, inplanes, planes, blocks, stride=1, baseWidth=26, scale=4, stype='normal', r=16, L=64):
        super(BasicStage, self).__init__()

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample, stype='stage', baseWidth=baseWidth, scale=scale))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, baseWidth=baseWidth, scale=scale))
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


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        print("baseWidth", self.baseWidth)
        print("scale", self.scale)
        self.stage1 = BasicStage(block, self.inplanes, 64, layers[0], baseWidth = self.baseWidth, scale=self.scale)
        self.inplanes = 256
        self.stage2 = BasicStage(block, self.inplanes, 128, layers[1], stride=2, baseWidth = self.baseWidth, scale=self.scale)
        self.inplanes = 512
        self.stage3 = BasicStage(block, self.inplanes, 256, layers[2], stride=2, baseWidth = self.baseWidth, scale=self.scale)
        self.inplanes = 1024
        self.stage4 = BasicStage(block, self.inplanes, 512, layers[3], stride=2, baseWidth = self.baseWidth, scale=self.scale)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
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


def res2net50(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)
    model = Res2Net(Bottle2neck, [5, 6, 12, 5], baseWidth = 26, scale = 4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model

def res2net50_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model

def res2net101_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth = 26, scale = 4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model

def res2net50_26w_6s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 6, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_6s']))
    return model

def res2net50_26w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_8s']))
    return model

def res2net50_48w_2s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_48w_2s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 48, scale = 2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_48w_2s']))
    return model

def res2net50_14w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_14w_8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 14, scale = 8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_14w_8s']))
    return model


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2net101_26w_4s(pretrained=True)
    model = model.cuda(0)
    print(model(images).size())
