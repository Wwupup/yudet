import torch
import torch.nn as nn
import cv2
from .nets.layers import PriorBox
from .nets.yunet import Yunet
from .nets.yuhead import Yuhead
from .losses.multiboxloss import MultiBoxLoss
from .src.utils import decode

class YuDetectNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Yunet()
        self.num_classes = cfg['model']['head']['num_classes']
        self.num_landmarks = cfg['model']['head']['num_landmarks']
        self.out_factor = (4 + self.num_landmarks * 2 + self.num_classes + 1)
        self.num_ratio = len(cfg['model']['anchor']['ratio'])
        self.head = Yuhead(
            in_channels=cfg['model']['head']['in_channels'],
            out_channels=[len(x) * self.out_factor * self.num_ratio for x in cfg['model']['anchor']['min_sizes']]
        )
        self.anchor_generator = PriorBox(
            min_sizes=cfg['model']['anchor']['min_sizes'],
            steps=cfg['model']['anchor']['steps'],
            clip=cfg['model']['anchor']['clip'],
            ratio=cfg['model']['anchor']['ratio']
        )
        self.criterion = MultiBoxLoss(
            num_classes=self.num_classes,
            iou_threshold=cfg['model']['loss']['overlap_thresh'],
            negpos_ratio=cfg['model']['loss']['neg_pos'],
            variance=cfg['model']['loss']['variance'],
            smooth_point=cfg['model']['loss']['smooth_point']
        )
        self.anchors_set = {}
        self.cfg = cfg
    
    def forward(self, x):
        self.img_size = x.shape[-2:]
        feats = self.backbone(x)
        outs = self.head(feats)
        head_data=[(x.permute(0, 2, 3, 1).contiguous()) for x in outs]
        head_data = torch.cat([o.view(o.size(0), -1) for o in head_data], dim=1)
        head_data = head_data.view(head_data.size(0), -1, self.out_factor)

        loc_data = head_data[:, :, 0 : 4 + self.num_landmarks * 2]
        conf_data = head_data[:, :, -self.num_classes - 1 : -1]
        iou_data = head_data[:,:, -1:]
        output = (loc_data, conf_data, iou_data)
        return output

    def get_anchor(self, img_shape=None):
        if img_shape is None:
            img_shape = self.img_size
        if len(img_shape) == 3:
            img_shape = img_shape[:2]
        if self.anchors_set.__contains__(img_shape):
            return self.anchors_set[img_shape]
        else:
            anchors = self.anchor_generator(img_shape)
            self.anchors_set[img_shape] = anchors
            return anchors

    def loss(self, predictions, targets):
        priors = self.get_anchor().cuda()
        loss_bbox_eiou, loss_iouhead_smoothl1, loss_lm_smoothl1, loss_cls_ce = \
            self.criterion(predictions, priors, targets)
        loss_bbox_eiou *= self.cfg['model']['loss']['weight_bbox']
        loss_iouhead_smoothl1 *= self.cfg['model']['loss']['weight_iouhead']
        loss_lm_smoothl1 *=  self.cfg['model']['loss']['weight_lmds']
        loss_cls_ce *= self.cfg['model']['loss']['weight_cls']
        return (loss_bbox_eiou, loss_iouhead_smoothl1, loss_lm_smoothl1, loss_cls_ce)

    def inference(self, img, scale, without_landmarks=True):
        if scale != 1.:
            img = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        priors = self.get_anchor(img.shape).cuda()
        h, w, _ = img.shape    
        img = torch.from_numpy(img).cuda()
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = img.float()
        loc, conf, iou = self(img)
        conf = torch.softmax(conf.squeeze(0), dim=-1)

        boxes = decode(loc.squeeze(0), priors.data, self.cfg['model']['loss']['variance'])
        box_dim = 4 if without_landmarks else (4 + self.num_landmarks * 2)
        boxes = boxes[:, :box_dim]
        boxes[:, 0::2] = boxes[:, 0::2] * w / scale
        boxes[:, 1::2] = boxes[:, 1::2] * h / scale
        cls_scores = conf.squeeze(0)[:, 1]
        iou_scores = iou.squeeze(0)[:, 0]

        iou_scores = torch.clamp(iou_scores, min=0., max=1.)
        scores = torch.sqrt(cls_scores * iou_scores)
        score_mask = scores > self.cfg['test']['confidence_threshold']
        boxes = boxes[score_mask]
        scores = scores[score_mask]
        _boxes = boxes[:, :4].clone()
        _boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        _boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        iou_thresh = self.cfg['test']['nms_threshold']
        keep_idx = cv2.dnn.NMSBoxes(
                    bboxes=_boxes.tolist(), 
                    scores=scores.tolist(), 
                    score_threshold=self.cfg['test']['confidence_threshold'], 
                    nms_threshold=iou_thresh, eta=1, 
                    top_k=self.cfg['test']['top_k']
        )
        if len(keep_idx) > 0:
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            dets = torch.cat([boxes, scores[:, None]], dim=-1)
            dets = dets[:self.cfg['test']['keep_top_k']]
        else:
            dets = torch.empty((0, box_dim + 1)).cuda()
        return dets