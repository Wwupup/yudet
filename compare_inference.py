import argparse
from datetime import datetime
import onnx
import onnxruntime
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
from model.nets.layers import PriorBox 
from math import ceil
from itertools import product as product


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def nms(dets, thresh, opencv_mode=True):
    if opencv_mode:
        _boxes = dets[:, :4].copy()
        scores = dets[:, -1]
        _boxes[:, 2] = _boxes[:, 2] - _boxes[:, 0]
        _boxes[:, 3] = _boxes[:, 3] - _boxes[:, 1]
        keep = cv2.dnn.NMSBoxes(
            bboxes=_boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=0.,
            nms_threshold=thresh,
            eta=1,
            top_k=5000
        )
        if len(keep) > 0:
            return keep.flatten()
        else:
            return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, -1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def resize_img(img, mode):
    if mode == 'ORIGIN':
        det_img, det_scale = img, 1.
    elif mode == "AUTO":
        assign_h = (img.shape[0] - 1 & (-32)) + 32
        assign_w = (img.shape[1] - 1 & (-32)) + 32
        det_img = np.zeros( (assign_h, assign_w, 3), dtype=np.uint8)
        det_img[:img.shape[0], :img.shape[1], :] = img
        det_scale = 1.
    else:
        if mode == "VGA":
            input_size = (640, 480)
        else:
            input_size = list(map(int, mode.split(',')))
        assert len(input_size) == 2
        x, y = max(input_size), min(input_size)
        if img.shape[1] > img.shape[0]:
            input_size = (x, y)
        else:
            input_size = (y, x)
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
        det_img[:new_height, :new_width, :] = resized_img

    return det_img, det_scale


def draw(img, bboxes, kpss, out_path):
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1,y1,x2,y2,score = bbox.astype(np.int32)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0) , 2)
        if kpss is not None:
            kps = kpss[i].reshape(-1, 2)
            for kp in kps:
                kp = kp.astype(np.int32)
                cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
        
    print('output:', out_path)
    cv2.imwrite(out_path, img) 

class Detector:
    def __init__(self, model_file=None, nms_thresh=0.5) -> None:
        self.model_file = model_file
        self.nms_thresh = nms_thresh
        assert osp.exists(self.model_file)
        model = onnx.load(model_file)
        onnx.checker.check_model(model) 
        self.session = onnxruntime.InferenceSession(self.model_file, None)
    def preprocess(self, img):
        pass
    def forward(self, img, score_thresh):
        pass
    def detect(self, img, score_thresh=0.5, mode="ORIGIN"):
        pass

class SCRFD(Detector):
    def __init__(self, model_file=None, nms_thresh=0.5) -> None:
        super().__init__(model_file, nms_thresh)
        self.taskname = 'scrfd'
        self.center_cache = {}
        self._init_vars()


    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs)==6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs)==9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs)==10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs)==15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True 

    def preprocess(self, img):
        pass
    def distance2bbox(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i%2] + distance[:, i]
            py = points[:, i%2+1] + distance[:, i+1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)    

    def forward(self, img, score_thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        t1 = datetime.now()
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name : blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            # If model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                #solution-1, c style:
                #anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                #for i in range(height):
                #    anchor_centers[i, :, 1] = i
                #for i in range(width):
                #    anchor_centers[:, i, 0] = i

                #solution-2:
                #ax = np.arange(width, dtype=np.float32)
                #ay = np.arange(height, dtype=np.float32)
                #xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                #anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                #solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                #print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= score_thresh)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = self.distance2kps(anchor_centers, kps_preds)
                #kpss = kps_preds
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list, datetime.now() - t1

    def detect(self, img, score_thresh=0.5, mode="ORIGIN", max_num=0, metric='default'):

        det_img, det_scale = resize_img(img, mode)

        scores_list, bboxes_list, kpss_list, t_forward = self.forward(det_img, score_thresh)

        scores = np.vstack(scores_list)
        # scores_ravel = scores.ravel()
        # order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        # pre_det = pre_det[order, :]
        keep = nms(pre_det, thresh=self.nms_thresh)
        det = pre_det[keep, :]
        if self.use_kps:
            # kpss = kpss[order,:,:]
            kpss = kpss[keep,:,:]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric=='max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss, t_forward

class YUNET(Detector):
    def __init__(self, model_file=None, nms_thresh=0.5) -> None:
        super().__init__(model_file, nms_thresh)
        self.taskname = 'yunet'
        self.anchor_fn = PriorBox(
            min_sizes=[[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
            steps=[8, 16, 32, 64],
            ratio=[1.],
            clip=False
        )
        self.priors_cache = None
    def preprocess(self, img):
        pass
    def forward(self, img, score_thresh, priors):
        t1 = datetime.now()
        img = img.astype(np.float32)
        img = np.transpose(img[None, ...], [0, 3, 1, 2]).copy()
        outs = self.session.run(None, {self.session.get_inputs()[0].name: img})
        loc, conf, iou = outs
        conf = softmax(conf.squeeze(0))
        boxes = self.decode(loc.squeeze(0), priors, variances=[0.1, 0.2])
        _, _, h, w = img.shape
        boxes[:, 0::2] = boxes[:, 0::2] * w
        boxes[:, 1::2] = boxes[:, 1::2] * h
        cls_scores = conf[:, 1]
        iou_scores = iou.squeeze(0)[:, 0]
        iou_scores = np.clip(iou_scores, a_min=0., a_max=1.)
        scores = np.sqrt(cls_scores * iou_scores)
        score_mask = scores > score_thresh
        boxes = boxes[score_mask]
        scores = scores[score_mask]
        return boxes, scores, datetime.now() - t1

    def detect(self, img, score_thresh=0.5, mode="ORIGIN"):
        det_img, det_scale = resize_img(img, mode)
        if mode == "ORIGIN" or mode == "AUTO":
            priors = self.anchor_fn(det_img.shape[:2]).numpy()
        else:

            if self.priors_cache is None:
                self.priors_cache = self.anchor_fn(det_img.shape[:2]).numpy()
            priors = self.priors_cache

        bboxes, scores, t_forward = self.forward(det_img, score_thresh, priors)
        bboxes /= det_scale
        pre_det = np.hstack((bboxes[:, :4], scores[:, None]))
        keep = nms(pre_det, self.nms_thresh)
        kpss = bboxes[keep, 4:]
        bboxes = pre_det[keep, :]
        return bboxes, kpss, t_forward

        
    def decode(self, loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        """        
        boxes = loc.copy()
        boxes[:, 0:2] = priors[:, 0:2] + boxes[:, 0:2] * variances[0] * priors[:, 2:4]
        boxes[:, 2:4] = priors[:, 2:4] * np.exp(boxes[:, 2:4] * variances[1])
        boxes[:, 0:2] -= boxes[:, 2:4] / 2
        boxes[:, 2:4] += boxes[:, 0:2]
        
        # landmarks
        if loc.shape[-1] > 4:
            boxes[:, 4::2] = priors[:, None, 0] + boxes[:, 4::2] * variances[0] * priors[:, None, 2]
            boxes[:, 5::2] = priors[:, None, 1] + boxes[:, 5::2] * variances[0] * priors[:, None, 3]
        return boxes
    
class YOLO5FACE(Detector):
    def __init__(self, model_file=None, nms_thresh=0.5) -> None:
        super().__init__(model_file, nms_thresh)

    def forward(self, img, score_thresh):
        t1 = datetime.now()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img[None, ...], [0, 3, 1, 2]).copy().astype(np.float32)
        img /= 255.
        outs = self.session.run(None, {self.session.get_inputs()[0].name: img})[0]
        outs = outs.squeeze(0)
        scores_mask = outs[:, 4] > score_thresh  # candidates
        outs = outs[scores_mask]

        outs[:, 15:] *= outs[:, 4:5]  # conf = obj_conf * cls_conf
        scores_mask = outs[:, 15] > score_thresh
        outs = outs[scores_mask]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = outs[:, :4].copy()
        box[:, 0] = outs[:, 0] - outs[:, 2] / 2  # top left x
        box[:, 1] = outs[:, 1] - outs[:, 3] / 2  # top left y
        box[:, 2] = outs[:, 0] + outs[:, 2] / 2  # bottom right x
        box[:, 3] = outs[:, 1] + outs[:, 3] / 2  # bottom right y

        boxes = np.hstack((box, outs[:, 5:-1]))
        scores = outs[:, -1]
        # dets = np.concatenate((box, outs[:, 5:], ), 1)
        return boxes, scores, datetime.now() - t1


    def detect(self, img, score_thresh=0.5, mode="ORIGIN"):
        assert mode == 'VGA' or mode == "640,640"
        det_img, det_scale = resize_img(img, mode)
        
        bboxes, scores, t_forward = self.forward(det_img, score_thresh)
        bboxes /= det_scale 
        pre_det = np.hstack((bboxes[:, :4], scores[:, None]))
        keep = nms(pre_det, self.nms_thresh)       
        kpss = bboxes[keep, 4:]
        bboxes = pre_det[keep, :]
        return bboxes, kpss, t_forward

class RETINAFACE(Detector):
    def __init__(self, model_file=None, nms_thresh=0.5) -> None:
        super().__init__(model_file, nms_thresh)
        self.priors_cache = None

    def anchor_fn(self, shape):
        min_sizes_cfg = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        clip = False
        shape = shape
        feature_maps = [[ceil(shape[0]/step), ceil(shape[1]/step)] for step in steps]

        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = min_sizes_cfg[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / shape[1]
                    s_ky = min_size / shape[0]
                    dense_cx = [x * steps[k] / shape[1] for x in [j + 0.5]]
                    dense_cy = [y * steps[k] / shape[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = np.array(anchors).reshape(-1, 4)
        if clip:
            output = output.clip(max=1, min=0)
        return output


    def decode(self, loc, priors, variances):
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes
    def decode_landms(self, pre, priors, variances):
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                    priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                    priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                    priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                    priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                    ), 1)
        return landms

    def forward(self, img, score_thresh, priors):
        t1 = datetime.now()
        img = img.astype(np.float32)
        img -= (104, 117, 123)
        img = np.transpose(img[None, ...], [0, 3, 1, 2]).copy()
        outs = self.session.run(None, {self.session.get_inputs()[0].name: img})
        loc, conf, landms = outs

        scores = conf.squeeze(0)[:, 1]
        boxes = self.decode(loc.squeeze(0), priors, variances=[0.1, 0.2])
        landms = self.decode_landms(landms.squeeze(0), priors, variances=[0.1, 0.2])

        boxes = np.concatenate((boxes, landms), 1)
        _, _, h, w = img.shape
        boxes[:, 0::2] = boxes[:, 0::2] * w
        boxes[:, 1::2] = boxes[:, 1::2] * h

        scores = conf.squeeze(0)[:, 1]

        score_mask = scores > score_thresh
        boxes = boxes[score_mask]
        scores = scores[score_mask]
        return boxes, scores, datetime.now() - t1

    def detect(self, img, score_thresh=0.5, mode="ORIGIN"):
        
        det_img, det_scale = resize_img(img, mode)
        if mode == "ORIGIN" or mode == "AUTO":
            priors = self.anchor_fn(det_img.shape[:2])
        else:

            if self.priors_cache is None:
                self.priors_cache = self.anchor_fn(det_img.shape[:2])
            priors = self.priors_cache

        bboxes, scores, t_forward = self.forward(det_img, score_thresh, priors)
        bboxes /= det_scale
        pre_det = np.hstack((bboxes[:, :4], scores[:, None]))
        keep = nms(pre_det, self.nms_thresh)
        kpss = bboxes[keep, 4:]
        bboxes = pre_det[keep, :]
        return bboxes, kpss, t_forward


def parse_args():
    parser = argparse.ArgumentParser(
        description='inference by ONNX')
    # for debug
    # parser.add_argument('--model_file', help='onnx model file path', default='/home/ww/projects/yudet/workspace/onnx/retinaface_mbnet0.25.onnx')
    # parser.add_argument('--eval', default=True, help='eval on widerface')

    parser.add_argument('model_file', help='onnx model file path')
    parser.add_argument('--eval', action='store_true', help='eval on widerface')
    parser.add_argument('--mode', type=str, default='ORIGIN', help='img scale. (640, 640) for VGA, choice=[VGA, ORIGIN, "number,number"]')
    parser.add_argument('--image', type=str, default='/home/ww/projects/yudet/data/widerface/WIDER_test/images/0--Parade/0_Parade_marchingband_1_9.jpg', help='image to detect')
    parser.add_argument('--nms_thresh', type=float, default=0.35, help='tresh to nms')

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    model_file = args.model_file
    assert osp.exists(model_file)
    if osp.basename(model_file).lower().startswith('scrfd'):
        prefix = 'scrfd'
        detector = SCRFD(model_file, nms_thresh=args.nms_thresh)
    elif osp.basename(model_file).lower().startswith('yunet'):
        prefix = 'yunet'
        detector = YUNET(model_file, nms_thresh=args.nms_thresh)
    elif osp.basename(model_file).lower().startswith('yolo5face'):
        prefix = 'yolo5face'
        detector = YOLO5FACE(model_file, nms_thresh=args.nms_thresh)
    elif osp.basename(model_file).lower().startswith('retinaface'):
        prefix = 'retinaface'
        detector = RETINAFACE(model_file, nms_thresh=args.nms_thresh)
    else:
        raise ValueError('Unknown detector!')
    if args.eval:
        from tools import get_testloader, evaluation
        widerface_root = './data/widerface/'
        split = 'val'
        testloader = get_testloader(
                mode='widerface',
                split=split,
                root=widerface_root
        )
        results = []
        for idx in tqdm(range(len(testloader))):
            img, mata = testloader[idx]
            bboxes, kpss, _ = detector.detect(img, score_thresh=0.02, mode=args.mode)#score_thresh=0.02 get the best mAP
            results.append({'pred': bboxes, 'mata': mata})

        evaluation(mode='widerface',
                    results=results,
                    results_save_dir='./images/results',
                    gt_root=osp.join(widerface_root, 'ground_truth'),
                    iou_tresh=0.5,
                    split=split)

    else:
        img = cv2.imread(args.image)
        epoch = 10
        cost_forward = 0
        for _ in range(epoch):        
            bboxes, kpss, _ = detector.detect(img, score_thresh=0.3, mode=args.mode)
        ta = datetime.now()
        for _ in range(epoch):        
            bboxes, kpss, t_forward = detector.detect(img, score_thresh=0.3, mode=args.mode)
            cost_forward += t_forward.total_seconds()
        tb = datetime.now()
        print('all cost:', (tb-ta).total_seconds()*1000 / epoch)
        print('forward cost:', cost_forward*1000 / epoch)

        draw(img, bboxes, kpss, out_path='./images/' + prefix + "_" + args.mode + osp.basename(args.image))


    
