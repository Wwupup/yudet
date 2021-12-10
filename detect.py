import argparse
import os
from numpy.core.fromnumeric import argsort
import yaml
import glob
import torch
import torch.backends.cudnn as cudnn
import cv2
import time
import numpy as np

from tools import Logger
from model import YuDetectNet
import glob

parser = argparse.ArgumentParser(description='Face and Landmark Detection')
parser.add_argument('--config', '-c', type=str, help='config to test')
parser.add_argument('--model', '-m', type=str, help='model path to test')
parser.add_argument('-t', '--target', type=str, help='image/image folder/video path')

def arg_initial(args):
    workfolder = os.path.dirname(os.path.dirname(args.model))
    cfg_list = glob.glob(os.path.join(workfolder, '*.yaml'))
    assert len(cfg_list) == 1, 'Can`t comfire config file!'
    with open(cfg_list[0], mode='r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    log_dir = os.path.join(workfolder, 'log')
    cfg['test']['log_dir'] = log_dir
    save_dir = os.path.join(workfolder, 'detect_results')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cfg['test']['save_dir'] = save_dir 
    cfg['test']['logger']['logfile'] = False
    cfg['test']['logger']['sysout'] = True
    assert os.path.exists(args.target)

    return cfg

def log_initial(cfg):
    return Logger(cfg, mode='test')

def detect_image(net, img_path, cfg):
    img = cv2.imread(img_path)
    det = net.inference(img, scale=1., without_landmarks=False)
    det = det.cpu().numpy()
    if len(det) == 0:
        print('Detect 0 taeget!')
        return
    scores = det[:, -1]
    det = det[:, :-1].astype(np.int32)
    for det, score in zip(det, scores):
        img = cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), color=(0, 0, 255), thickness=1)
        img = cv2.putText(img, f"{score:4f}", (det[0], det[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        ldms_num = int((det.shape[-1] - 4) / 2)
        for i in range(ldms_num):
            cv2.circle(img, (det[4 + 2 * i], det[5 + 2 * i]), 2, (255, 255, 0), thickness=5)
    save_path = os.path.join(cfg['test']['save_dir'], os.path.basename(img_path))
    cv2.imwrite( save_path, img)
    print(f'Detect {0 if len(det.shape) == 1 else det.shape[0]} target, Save img to {save_path}')

def detect_video(net, video_path, cfg):
    cap = cv2.VideoCapture(video_path)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_path = os.path.join(cfg['test']['save_dir'], os.path.basename(video_path))
    video_writer = cv2.VideoWriter(
                save_path, 
                cv2.VideoWriter_fourcc('M','P','E','G'),
                fps,
                size
    )
    while(True):
        ret, frame = cap.read()
        if ret:
            det = net.inference(frame, scale=1., without_landmarks=False).cpu().numpy()
            scores = det[:, -1]
            det = det[:, :-1].astype(np.int32)
            for det, score in zip(det, scores):
                frame = cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), color=(0, 0, 255), thickness=1)
                frame = cv2.putText(frame, f"{score:4f}", (det[0], det[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                for i in range((det.shape[1] - 5) / 2):
                    cv2.circle(frame, (det[4 + 2 * i], det[5 + 2 * i]), 2, (255, 255, 0), thickness=5)
            video_writer.write(frame)
        else:
            break
    cap.release()
    video_writer.release()
    print(f'Save video to {save_path}')


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

    target = args.target
    img_paths = []
    if os.path.isdir(target):
        img_paths = glob.glob(os.path.join(target, '*.jpg'))
        img_paths.append(glob.glob(os.path.join(target, '*.jpeg'))) 
        img_paths.append(glob.glob(os.path.join(target, '*.png')))         
        img_paths.append(glob.glob(os.path.join(target, '*.mp4')))
    else:
        img_paths.append(target)

    print(f'{len(img_paths)} files to be detected...')
    for img_path in img_paths:
        filename, tp = os.path.splitext(os.path.basename(img_path))
        if tp.lower() in ('.jpg', '.jpeg', '.png'):
            detect_image(net, img_path, cfg)
        elif tp.lower() in ('.mp4'):
            detect_video(net, img_path, cfg)
        else:
            print('Unsupport file!')


if __name__ == "__main__":
    main()