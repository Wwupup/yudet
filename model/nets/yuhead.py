from typing import List
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv4layerBlock


class Yuhead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert len(in_channels) == len(out_channels)
        self.head = nn.ModuleList(
            [Conv4layerBlock(in_c, out_c, withBNRelu=False) for \
                in_c, out_c in zip(in_channels, out_channels)]
        )
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        assert isinstance(feats, List)
        outs = []
        up = self.head[-1].conv1(feats[-1])
        out = self.head[-1].conv2(up)
        outs.append(out)
        for i in range(len(feats) - 2, -1, -1):
            up = F.interpolate(
                up, 
                size=[feats[i].size(2), feats[i].size(3)], 
                mode="nearest"
            )
            up = self.head[i].conv1(feats[i] + up)
            out = self.head[i].conv2(up)
            outs.insert(0, out)

        return outs
        



