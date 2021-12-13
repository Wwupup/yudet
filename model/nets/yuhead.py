from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv4layerBlock, ConvDPUnit


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
        
class Yuhead_PAN(Yuhead):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

    def forward(self, feats):
        assert isinstance(feats, List)
        num_feats = len(feats)
        fpn_feats = []

        # up-bottom
        up = self.head[-1].conv1(feats[-1])
        fpn_feats.append(up)
        for i in range(num_feats - 2, -1, -1):
            up = F.interpolate(
                up, 
                size=[feats[i].size(2), feats[i].size(3)], 
                mode="nearest"
            )
            up = self.head[i].conv1(feats[i] + up)
            fpn_feats.insert(0, up)

        # bottom-up
        outs = []
        outs.append(self.head[0].conv2(fpn_feats[0]))
        for i in range(1, num_feats):
            down = F.interpolate(
                fpn_feats[i - 1], 
                size=[fpn_feats[i].size(2), fpn_feats[i].size(3)], 
                mode="nearest"
            )
            outs.append(self.head[i].conv2(fpn_feats[i] + down))

        return outs

class Yuhead_double(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert len(in_channels) == len(out_channels)

        self.mix = nn.ModuleList(
            [ConvDPUnit(in_c, in_c, withBNRelu=True) for \
                in_c in in_channels]
        )
        self.head_cls = nn.ModuleList(
            [ConvDPUnit(in_c, int(out_c / 17) * 2, withBNRelu=False) for \
                in_c, out_c in zip(in_channels, out_channels)]
        )
        self.head_reg = nn.ModuleList(
            [ConvDPUnit(in_c, int(out_c / 17) * 15, withBNRelu=False) for \
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
        outs_cls = []
        outs_reg = []
        up = self.mix[-1](feats[-1])
        outs_cls.append(self.head_cls[-1](up))
        outs_reg.append(self.head_reg[-1](up))
        for i in range(len(feats) - 2, -1, -1):
            up = F.interpolate(
                up, 
                size=[feats[i].size(2), feats[i].size(3)], 
                mode="nearest"
            )
            up = self.mix[i](feats[i] + up)
            outs_cls.insert(0, self.head_cls[i](up))
            outs_reg.insert(0, self.head_reg[i](up))

        outs = []
        for cls, reg in zip(outs_cls, outs_reg):
            n, c, h, w = cls.shape

            cls_data = cls.permute(0, 2, 3, 1).view(n, h, w, -1, 2)
            reg_data = reg.permute(0, 2, 3, 1).view(n, h, w, -1, 15)
            out = torch.cat([reg_data[..., :-1], cls_data[..., :], reg_data[..., -1:]], dim=-1)
            outs.append(out.view(n, h, w, -1).permute(0, 3, 1, 2).contiguous())
        return outs