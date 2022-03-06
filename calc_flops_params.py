import argparse
import os
import yaml
import glob
import torch
import torch.backends.cudnn as cudnn
import time

from torchinfo import summary

from model import YuDetectNet

parser = argparse.ArgumentParser(description='Face and Landmark Detection')
parser.add_argument('config', type=str, help='config to test')

def arg_initial(args):
    with open(args.config, mode='r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def main():
    args = parser.parse_args()
    cfg = arg_initial(args)
    net = YuDetectNet(cfg).cuda()
    print("Torchinfo:")
    print(summary(net, (1, 3, 320, 320),col_names=["kernel_size", "output_size", "num_params", "mult_adds"],))
    # print('Torchstat:')
    # stat(net,(3,320,320))


    


if __name__ == '__main__':
    main()



"""
FLOPs = 2 * MACs = 2 * MAdds
NVIDIA paper : ICLR2017

"""