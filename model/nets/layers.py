import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvDPUnit(nn.Module):
    def __init__(self, in_channels, out_channels, withBNRelu=True):
        super(ConvDPUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True, groups=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True, groups=out_channels)
        self.withBNRelu = withBNRelu
        if withBNRelu:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.withBNRelu:
            x = self.bn(x)
            x = F.relu(x, inplace=True)
        return x

class Conv_head(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Conv_head, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 2, 1, bias=True, groups=1)
        self.conv2 = ConvDPUnit(mid_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        return x

class Conv4layerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, withBNRelu=True):
        super(Conv4layerBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvDPUnit(in_channels, in_channels, True)
        self.conv2 = ConvDPUnit(in_channels, out_channels, withBNRelu)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



from itertools import product as product
class PriorBox(object):
    def __init__(self, min_sizes, steps, clip, ratio):
        super(PriorBox, self).__init__()
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.ratio = ratio
    def __call__(self, image_size):
        feature_map_2th = [int(int((image_size[0] + 1) / 2) / 2),
                                int(int((image_size[1] + 1) / 2) / 2)]
        feature_map_3th = [int(feature_map_2th[0] / 2),
                                int(feature_map_2th[1] / 2)]
        feature_map_4th = [int(feature_map_3th[0] / 2),
                                int(feature_map_3th[1] / 2)]
        feature_map_5th = [int(feature_map_4th[0] / 2),
                                int(feature_map_4th[1] / 2)]
        feature_map_6th = [int(feature_map_5th[0] / 2),
                                int(feature_map_5th[1] / 2)]

        feature_maps = [feature_map_3th, feature_map_4th,
                             feature_map_5th, feature_map_6th]
        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    cx = (j + 0.5) * self.steps[k] / image_size[1]
                    cy = (i + 0.5) * self.steps[k] / image_size[0]
                    for r in self.ratio:
                        s_ky = min_size / image_size[0]
                        s_kx = r * min_size / image_size[1]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output