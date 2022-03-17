from distutils.command.config import config
from time import time
import cv2
import numpy as np
import onnx
import onnxruntime
from model.nets.layers import PriorBox 
import os
import torch
from model.yudet import YuDetectNet
import yaml

#inference for default config/yufacedet.yaml
def inference(outs, img, anchor_fn):
    variances = [0.1, 0.2]
    box_dim = 4 #14 if with landmark else 4
    confidence_threshold = 0.3
    nms_threshold = 0.3
    top_k = 5000
    keep_topk = 750


 
    priors = anchor_fn(img.shape[:2])
    # conf = softmax(conf.squeeze(0))
    t1 = time()
    loc, conf, iou = outs    
    conf = torch.softmax(torch.from_numpy(conf.squeeze(0)), dim=-1).numpy()
    boxes = decode_numpy(loc.squeeze(0), priors.numpy(), variances=variances)
    boxes = boxes[:, :box_dim]
    h, w, _ = img.shape
    boxes[:, 0::2] = boxes[:, 0::2] * w
    boxes[:, 1::2] = boxes[:, 1::2] * h
    cls_scores = conf[:, 1]
    iou_scores = iou.squeeze(0)[:, 0]
    iou_scores = np.clip(iou_scores, a_min=0., a_max=1.)
    scores = np.sqrt(cls_scores * iou_scores)
    score_mask = scores > confidence_threshold
    boxes = boxes[score_mask]
    scores = scores[score_mask]
    _boxes = boxes[:, :4].copy()
    _boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    _boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    keep_idx = cv2.dnn.NMSBoxes(
                bboxes=_boxes.tolist(), 
                scores=scores.tolist(), 
                score_threshold=confidence_threshold, 
                nms_threshold=nms_threshold,
                eta=1, 
                top_k=top_k
    )
    if len(keep_idx) > 0:
        keep_idx = keep_idx.reshape(-1)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        dets = np.concatenate([boxes, scores[:, None]], axis=-1)
        dets = dets[:keep_topk]
    else:
        dets = np.empty((0, box_dim + 1))
    return dets, time() - t1


def load_onnx(onnx_path):
    net = onnx.load(onnx_path)
    onnx.checker.check_model(net) 
    ort_session = onnxruntime.InferenceSession(onnx_path)
    return ort_session  

def load_pytorch_model(config, model):
    with open(config, mode='r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    torch.set_grad_enabled(False)
    net = YuDetectNet(cfg)
    net.load_state_dict(torch.load(model))
    net.eval()
    return net

def preprocess(img):
    input = img.astype(np.float32)
    input = np.transpose(input[None, ...], [0, 3, 1, 2]).copy()
    return input

def decode_numpy(loc, priors, variances):
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

def to_numpy(tensor): 
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis=-1).reshape(-1, 1)
    return softmax_x

def draw_image(dets, img):
    scores = dets[:, -1]
    dets = dets[:, :-1].astype(np.int32)
    for det, score in zip(dets, scores):
        img = cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), color=(0, 0, 255), thickness=1)
        # img = cv2.putText(img, f"{score:4f}", (det[0], det[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        ldms_num = int((det.shape[-1] - 4) / 2)
        for i in range(ldms_num):
            cv2.circle(img, (det[4 + 2 * i], det[5 + 2 * i]), 2, (255, 255, 0), thickness=1)
    return img


def main():
    img_path = "/home/ww/projects/yudet/data/widerface/WIDER_test/images/6--Funeral/6_Funeral_Funeral_6_17.jpg"
    onnx_path = "/home/ww/projects/yudet/workspace/onnx/best_rebuild_dynamic.onnx"
    config_path = '/home/ww/projects/yudet/workspace/facenvive/yufacedet.yaml'
    model_path = '/home/ww/projects/yudet/workspace/facenvive/weights/best_rebuild.pth'
    anchor_fn = PriorBox(
        min_sizes=[[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
        steps=[8, 16, 32, 64],
        ratio=[1.],
        clip=False
    )
    img = cv2.imread(img_path)
    img = cv2.resize(img, (480, 640))
    input = preprocess(img)

    # onnx inference
    ort_session = load_onnx(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input}


    # pytorch inference
    # torch_session = load_pytorch_model(config_path, model_path)
    # outs_torch = torch_session(torch.from_numpy(input))

    # np.testing.assert_allclose(to_numpy(outs_torch), outs, rtol=1e-03, atol=1e-05)

    epoch = 1000
    a, b = 0, 0
    for _ in range(epoch):
        t1 = time()
        outs = ort_session.run(None, ort_inputs)
        a += time() - t1
        dets, t3 = inference(outs, img, anchor_fn)
        b += t3
    print(f'Shape: {img.shape} Epoch 1000:')
    print(f"Forward time / Postprocess Time = {a / b}")
    print(f"Forward time / Total Time = {a / (a + b)}")
    print(f"Postprocess time / Total Time = {b / (a + b)}")
    print(f'Total Time: {a + b}s, achieve {epoch / (a + b)} fps')
    print(f'Forward achieve {epoch / a} fps')
    print(f'Postprocess achieve {epoch / b} fps')

    img_result = draw_image(dets, img)
    h, w, _ = img.shape
    save_path = os.path.join('./images', f"demo_{h}_{w}.jpg")
    cv2.imwrite(save_path, img_result)


if __name__ == "__main__":
    main()
