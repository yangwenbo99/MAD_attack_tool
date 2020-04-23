#!/bin/python3

import os
import time
import scipy.stats
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import cv2
import argparse
from multiprocessing import Pool
from matplotlib import pyplot as plt


import madatk
from madatk import MADGenerator
import pytorch_ssim
import pytorch_msssim
import mse
from dir_loader import Dirloader, ListLoader

def get_fun(name : str):
    name = name.upper()
    if name == 'MSE':
        return mse.mse
    elif name == 'SSIM':
        return pytorch_ssim.ssim
    elif name == 'MSSSIM':
        # return pytorch_msssim.msssim
        return pytorch_msssim.MS_SSIM(data_range=1.0, nonnegative_ssim=True)

def read_img(filename: str):
    img= Image.open(filename)
    img = transform(img)
    img = img.unsqueeze(0)
    return img

transform = transforms.ToTensor()
transform_back = transforms.ToPILImage()

def parse_config():
    parser = argparse.ArgumentParser(epilog=\
            'Functions may be one of the followings:\n' +
            '\tmse\n' +
            '\tssim\n' +
            '\tmsssim\n'
            )

    parser.add_argument("fun1",
            default=0.1, type=str)
    parser.add_argument("fun2",
            default=0.1, type=str)
    parser.add_argument("in_dir", default='unconstrained',
            help='Directory of input files', type=str)
    parser.add_argument("ref_dir", default='unconstrained',
            help='Directory of reference files', type=str)
    parser.add_argument("out_dir", default='unconstrained',
            help='Directory to output', type=str)
    parser.add_argument("-l", "--filelist",
            help='If the relationship between ref and in is provided in filem' + \
                 'use this argument',
            default=None)
    parser.add_argument("--nocuda", default=False, action="store_true")
    parser.add_argument("--alpha1p",
            default=0.1,
            help='Initial learning rate for increasing the first function',
            type=float)
    parser.add_argument("--alpha1n",
            default=-0.1,
            help='Initial learning rate for decreasing the first function',
            type=float)
    parser.add_argument("--alpha2p",
            default=0.1,
            help='Initial learning rate for increasing the second function',
            type=float)
    parser.add_argument("--alpha2n",
            default=-0.1,
            help='Initial learning rate for decreasing the second function',
            type=float)
    parser.add_argument("--lmd",
            default=2.0,
            type=float)
    parser.add_argument("-i", "--iter",
            default=100,
            type=int)

    return parser.parse_args()

def perform(cfg):
    if cfg.filelist:
        with open(cfg.filelist, 'r') as f:
            l = [s.split() for s in f.readlines()]
            loader = ListLoader(cfg.in_dir, cfg.ref_dir, l)
    else:
        loader = Dirloader(cfg.in_dir, cfg.ref_dir)
    input_path = Path(cfg.in_dir)
    refdir_path = Path(cfg.ref_dir)
    output_path = Path(cfg.out_dir)
    device = torch.device( "cuda:0" if torch.cuda.is_available() and not cfg.nocuda else "cpu")
    fun1 = get_fun(cfg.fun1)
    fun2 = get_fun(cfg.fun2)

    path_front = str(input_path)
    for fullpath, ref_fullpath in loader:
        print(fullpath)
        output_filepath = \
                output_path / Path(str(fullpath)[len(path_front)+1:])
        output_filepath = output_filepath.with_suffix('.png')

        img = read_img(fullpath).to(device)
        ref = read_img(ref_fullpath).to(device)

        gen = MADGenerator(img, ref, fun1, fun2, lmd=2, max_iter=cfg.iter)
        gen.calculate()

        print('{:30}'.format(str(fullpath)))

        for i in ('ps', 'ns', 'sp', 'sn'):
            on = output_filepath.with_name(
                    output_filepath.stem + '_' + i + '.png')
            transform_back(gen[i][0].cpu()).save(on)
            score1 = fun1(gen[i], ref).cpu().numpy()
            score2 = fun2(gen[i], ref).cpu().numpy()
            print('    {:2} {:.5} {:.5}'.format(i, score1, score2))


if __name__ == "__main__":
    cfg = parse_config()
    perform(cfg)

