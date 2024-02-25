import torch
import torch.nn as nn


__all__ = ["SalientBlock", "GroupSalientBlock", "SEBlock", "CBAMBlock"]

class SalientBlock(nn.Module):
    def __init__(self, in_planes):
        super(SalientBlock, self).__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        size_n, _, size_h, size_w = list(x.size())
        w = self.globalAvgPool(x)
        w = torch.mean(w*x, 1).view((size_n, 1, size_h, size_w))
        spatial_w = self.sigmoid(self.bn(w))
        out = spatial_w * x
        return out

# get 1 attention map by every k feature maps. 
# If in_planes=256, channels_per_maps=16, we could get 256/16=16 attention maps.
class GroupSalientBlock(nn.Module):
    CHANNELS_PER_MAP = 32
    def __init__(self, in_planes, channels_per_map=CHANNELS_PER_MAP):
        super(GroupSalientBlock, self).__init__()
        assert channels_per_map>in_planes or in_planes%channels_per_map==0
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.groups = in_planes//channels_per_map if in_planes//channels_per_map>0 else 1
        self.bn = nn.BatchNorm2d(self.groups)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        size_n, size_c, size_h, size_w = list(x.size())
        ca_w = self.globalAvgPool(x).view((size_n, self.groups, -1, 1, 1))
        x = x.view((size_n, self.groups, -1, size_h, size_w))
        # w = torch.mean(w*x, 2).view((size_n, self.groups, size_h, size_w))
        ca_w = torch.mean(ca_w*x, 2)
        spatial_w = self.sigmoid(self.bn(ca_w)).view((size_n, self.groups, 1, size_h, size_w))
        out = spatial_w * x
        out = out.view((size_n, size_c, size_h, size_w))
        return out


class SEBlock(nn.Module):
    reduction_ratio = 16
    def __init__(self, in_planes, ratio = reduction_ratio):
        super(SEBlock, self).__init__()
        self.in_planes = in_planes
        self.ratio = ratio
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.linear(in_planes, in_planes//self.ratio)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.linear(in_planes//self.ratio, in_planes)
        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        n, c, h, w = x.size()
        y = self.GAP(x).view((n,c))
        y = self.relu1(self.fc1(y))
        # y = self.Sigmoid(self.fc2(y)).view((n,c,1,1))
        y = self.Sigmoid(self.fc2(y))
        return y*x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class CBAMBlock(nn.Module):
    def __init__(self, in_planes):
        super(CBAMBlock, self).__init__()
        kernel_size = 7
        self.spatial_pool = ChannelPool()
        self.se_attention = SEBlock(in_planes)
        self.sa_conv = nn.Sequential(
                                    nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid()
                                    )
    
    def forward(self, x):
        n, c, h, w = x.size()
        se_out = self.se_attention(x)
        sa_map = self.sa_conv(self.spatial_pool(se_out))
        return sa_map*se_out


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_planes):
        super(SpatialAttentionBlock, self).__init__()
        self.sa_ops = nn.Sequential(
                                    nn.Conv2d(in_planes,  1, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid()
                                    )
    
    def forward(self, x):
        sa_map = self.sa_ops(x)
        return sa_map*x
