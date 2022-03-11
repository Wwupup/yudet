import argparse
import os
import yaml
import sys
import torch
import cv2
import time
import numpy as np
import onnx
import onnxruntime

from model import YuDetectNet

parser = argparse.ArgumentParser(description='Face and Landmark Detection')
parser.add_argument('--config', '-c', default="/home/ww/projects/yudet/workspace/facenvive/yufacedet.yaml", type=str, help='config to test')
parser.add_argument('--model', '-m', default="/home/ww/projects/yudet/workspace/facenvive/weights/best_rebuild.pth", type=str, help='model weights path')
parser.add_argument('--tag', '-t', type=str, help='tag to mark weight')

def arg_initial(args):
    with open(args.config, mode='r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    return cfg

def main():
    args = parser.parse_args()
    cfg = arg_initial(args)

    torch.set_grad_enabled(False)
    net = YuDetectNet(cfg)
    net.load_state_dict(torch.load(args.model), strict=True)
    net.eval()

    img = torch.randn(1, 3, 320, 320, requires_grad=False)
    input_names = ['input']
    output_names = ['loc', 'conf', 'iou']
    output_path = os.path.abspath(os.path.join('./workspace', 'onnx', f'{os.path.basename(args.model[:-4])}.onnx')) 

    print(f'Export:\n{args.model}\nTo onnx:\n{output_path}')
    torch.onnx.export(
        model=net,
        args=img,
        f=output_path, 
        input_names=input_names,
        output_names=output_names,        
        export_params=True,
        verbose=False,
        do_constant_folding=True,
        dynamic_axes=None,
        opset_version=11)

    print('Test model:')
    net_onnx = onnx.load(output_path)
    onnx.checker.check_model(net_onnx)
    ort_session = onnxruntime.InferenceSession(output_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    
    # net_opencv = cv2.dnn.readNetFromONNX(output_path)

    epoch = 1000
    ort_t_1 = time.time()
    for i in range(epoch):
        ort_outs = ort_session.run(None, ort_inputs)
    ort_t_2 = time.time()
    
    torch_t_1 = time.time()
    for i in range(epoch):
        torch_outs = net(img)
    torch_t_2 = time.time()

    # opencv_t_1 = time.time()
    # for i in range(epoch):
    #     net_opencv.setInput(img)
    #     opencv_outs = net_opencv.forward()
    # opencv_t_2 = time.time()

    # print(f"Loop {epoch}\ntorch time:{torch_t_2 - torch_t_1}\nort time: {ort_t_2 - ort_t_1}\nopencv time: {opencv_t_2 - opencv_t_1}")
    for torch_out, ort_out in zip(torch_outs, ort_outs):
        np.testing.assert_allclose(to_numpy(torch_out), ort_out, rtol=1e-03, atol=1e-05)

    # for torch_out, ort_out in zip(torch_outs, ort_outs):
    #     np.testing.assert_allclose(to_numpy(torch_out), opencv_outs, rtol=1e-03, atol=1e-05)  

    print('Successful!')
    
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    main()
    