import glob
import os
import time

import cv2
import numpy as np
import torch
# import xlwt

from .modules.generator import Generator

device = 'cuda:0'
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.png"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
    data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    return data

def input_setup(data_vi, data_ir, index):
    padding = 0
    sub_ir_sequence = []
    sub_vi_sequence = []
    _ir = imread_gray(data_ir[index])                                                   # 读取单通道图像
    _vi = imread_gray(data_vi[index])
    input_ir = (_ir - 127.5) / 127.5
    input_ir = np.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_ir.shape
    input_ir = input_ir.reshape([w, h, 1])
    input_vi = (_vi - 127.5) / 127.5
    input_vi = np.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_vi.shape
    input_vi = input_vi.reshape([w, h, 1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir = np.asarray(sub_ir_sequence)
    train_data_vi = np.asarray(sub_vi_sequence)
    return train_data_ir, train_data_vi

def input_frame(vi, ir):
    padding = 0

    input_ir = (ir - 127.5) / 127.5
    input_ir = np.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_ir.shape
    input_ir = input_ir.reshape([w, h, 1])

    input_vi = (vi - 127.5) / 127.5
    input_vi = np.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_vi.shape
    input_vi = input_vi.reshape([w, h, 1])

    return input_vi, input_ir

def imread_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)                                        # 读取单通道图像
    # img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img[:, :]

def makepath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def all(vi, ir):
    g = Generator().to(device)

    weights = torch.load("H:\gaoshanshaobing_v0.6\ATGAN\checkpoint\epoch_72\model-72.pt")
    # weights = torch.load(r"G:\gaoshanshaobing\ATGAN\checkpoint\epoch_72\model-72.pt")
    g.load_state_dict(weights)                                                                                                  # 载入权重
    g.eval()

    train_data_ir, train_data_vi = input_frame(vi, ir)

    train_data_ir = np.expand_dims(train_data_ir, axis=0)
    train_data_vi = np.expand_dims(train_data_vi, axis=0)

    train_data_ir = train_data_ir.transpose([0, 3, 1, 2])
    train_data_vi = train_data_vi.transpose([0, 3, 1, 2])

    train_data_ir = torch.tensor(train_data_ir).float().to(device)
    train_data_vi = torch.tensor(train_data_vi).float().to(device)

    result = g(train_data_ir, train_data_vi)
    result = np.squeeze(result.cpu().numpy() * 127.5 + 127.5).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(result)

    return result

# if __name__ == '__main__':
#     dataset_name=["INO", 'M3', 'MFNet', 'RoadScene', 'TNO']
#     for d in range(1, len(dataset_name)):
#         for e in range(72, 73):
#             print("test epoch" + str(e) + ' on the '+dataset_name[d]+'\n')
#             all(i=e, dataset=dataset_name[d])
