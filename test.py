import argparse
import os
import yaml
import glob
from tqdm import tqdm
import sys
import torch
import torch.backends.cudnn as cudnn
import cv2
import time
import numpy as np

from tools import Logger
from model import YuDetectNet
from tools import widerface_evaluation
from tools import WIDERFace

parser = argparse.ArgumentParser(description='Face and Landmark Detection')
parser.add_argument('--config', '-c', type=str, help='config to test')
parser.add_argument('--model', '-m', type=str, help='model path to test')


def arg_initial(args):
    workfolder = os.path.dirname(os.path.dirname(args.model))
    cfg_list = glob.glob(os.path.join(workfolder, '*.yaml'))
    assert len(cfg_list) == 1, 'Can`t comfire config file!'
    with open(cfg_list[0], mode='r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    log_dir = os.path.join(workfolder, 'log')
    cfg['test']['log_dir'] = log_dir
    save_dir = os.path.join(workfolder, 'results')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cfg['test']['save_dir'] = save_dir    
    # with open(os.path.join(workfolder, os.path.basename(args.config)), mode='w', encoding='utf-8') as f:
    #     yaml.safe_dump(cfg, f)
    return cfg

def log_initial(cfg):
    return Logger(cfg, mode='test')

def main():
    args = parser.parse_args()
    cfg = arg_initial(args)
    logger = log_initial(cfg)
    torch.set_grad_enabled(False)

    logger.info(f'Loading model from {args.model}')
    net = YuDetectNet(cfg)
    net.load_state_dict(torch.load(args.model))
    net.eval()
    net.cuda()
    cudnn.benchmark = True
    widerface = WIDERFace(cfg['test']['dataset']['root'], cfg['test']['dataset']['split'])
    scales = [0.25, 0.50, 0.75, 1.25, 1.50, 1.75, 2.0] if cfg['test']['multi_scale'] else [1.]
    logger.info(f'Performing testing with scales: {str(scales)}, conf_threshold: {cfg["test"]["confidence_threshold"]}')
    for idx in tqdm(range(len(widerface))):
    # for idx in range(len(widerface)):
        img, event, name = widerface[idx] # img_subpath = '0--Parade/XXX.jpg'
        available_scales = get_available_scales(img.shape[0], img.shape[1], scales)
        dets = torch.empty((0, 5)).cuda()
        for available_scale in available_scales:
            det = net.inference(img, available_scale)
            dets = torch.cat([dets, det], dim=0)
        save_res(dets.cpu(), event, name, save_path=os.path.join(cfg['test']['save_dir'], event))
        # draw(img, dets.cpu().numpy(), idx)

    logger.info('Evaluating:')
    sys.stdout = logger
    widerface_evaluation(cfg['test']['save_dir'], os.path.join(cfg['test']['dataset']['root'], './ground_truth'))

def save_res(dets, event, name, save_path):
    txt_name = name[:-4]+'.txt'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, txt_name), 'w') as f:
        f.write('{}\n'.format('/'.join([event, name])))
        f.write('{}\n'.format(dets.shape[0]))
        for k in range(dets.shape[0]):
            xmin = dets[k, 0]
            ymin = dets[k, 1]
            xmax = dets[k, 2]
            ymax = dets[k, 3]
            score = dets[k, 4]
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            f.write(f'{torch.floor(xmin):.1f} {torch.floor(ymin):.1f} {torch.ceil(w):.1f} {torch.ceil(h):.1f} {score:.3f}\n')

def draw(img, pred, idx = 0):
    scores = pred[:, -1]
    dets = pred[:, :-1].astype(np.int32)
    for det, score in zip(dets, scores):
        img = cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), color=(0, 0, 255), thickness=1)
        # img = cv2.putText(img, f"{score:4f}", (det[0], det[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        # for i in range(4):
        #     cv2.circle(img, (det[4 + 2 * i], det[5 + 2 * i]), 2, (255, 255, 0), thickness=5)
    save_dir = "./results"
    cv2.imwrite( os.path.join(save_dir, f"{idx}_{score:.4f}.jpg"), img)

def get_available_scales(h, w, scales):
    smin = min(h, w)
    available_scales = []
    for scale in scales:
        if int(smin * scale) >= 64:
            available_scales.append(scale)
    return available_scales

if __name__ == "__main__":
    main()
