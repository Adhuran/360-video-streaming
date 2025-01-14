import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url


__all__ = ['SalientResNet18', 'SalientResNet34', 'SalientResNet50', 'SalientResNet101',
           'SalientResNet152']


# model_urls = {
#     'salientnet18': 'https://download.pytorch.org/models/salientnet18-5c106cde.pth',
#     'salientnet34': 'https://download.pytorch.org/models/salientnet34-333f7ec4.pth',
#     'salientnet50': 'https://download.pytorch.org/models/salientnet50-19c8e357.pth',
#     'salientnet101': 'https://download.pytorch.org/models/salientnet101-5d3b4d8f.pth',
#     'salientnet152': 'https://download.pytorch.org/models/salientnet152-b121ed2d.pth',
#     'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#     'wide_salientnet50_2': 'https://download.pytorch.org/models/wide_salientnet50_2-95faca4d.pth',
#     'wide_salientnet101_2': 'https://download.pytorch.org/models/wide_salientnet101_2-32ee1156.pth',
# }


class SalientBlock(nn.Module):
    def __init__(self, in_planes):
        super(SalientBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # w = torch.sigmoid(self.bn(self.conv(x)))
        #out = (1+w) * x # boradcasting mul
        #out = w * x
        size_n, _, size_h, size_w = list(x.size())
        w = self.globalAvgPool(x)
        #w = w*x
        w = torch.mean(w*x, 1).view((size_n, 1, size_h, size_w))
        spatial_w = self.sigmoid(self.bn(w))
        out = spatial_w * x
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.Salient = SalientBlock(planes)     # add by cdt, using the SalientBlock to attentation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.Salient(out)             # add by cdt, using the SalientBlock to attentation

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.Salient = SalientBlock(planes * self.expansion)     # add by cdt, using the SalientBlock to attentation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.Salient(out)             # add by cdt, using the SalientBlock to attentation
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class salientnet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(salientnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _salientnet(arch, block, layers, pretrained, progress, **kwargs):
    model = salientnet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def SalientResNet18(pretrained=False, progress=True, **kwargs):
    return _salientnet('salientnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def SalientResNet34(pretrained=False, progress=True, **kwargs):
    return _salientnet('salientnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def SalientResNet50(pretrained=False, progress=True, **kwargs):
    return _salientnet('salientnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def SalientResNet101(pretrained=False, progress=True, **kwargs):
    return _salientnet('salientnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def SalientResNet152(pretrained=False, progress=True, **kwargs):
    return _salientnet('salientnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


# def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     return _salientnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)


# def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     return _salientnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)


# def wide_salientnet50_2(pretrained=False, progress=True, **kwargs):
#     kwargs['width_per_group'] = 64 * 2
#     return _salientnet('wide_salientnet50_2', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)


# def wide_salientnet101_2(pretrained=False, progress=True, **kwargs):
#     kwargs['width_per_group'] = 64 * 2
#     return _salientnet('wide_salientnet101_2', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)
