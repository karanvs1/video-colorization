from matplotlib import lines
from utils import *
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt

class VCdataset_train(Dataset):
    def __init__(self):
        self.data_path = 'sample dataset'
        self.frame_list = os.path.join(self.data_path, 'sample_davis_train.txt')
        self.file = open(self.frame_list, 'r')
        self.lines = self.file.read()
        self.frame_list = self.lines.split('\n')

    def __getitem__(self, index):
        images = []
        for i in range(index, index + 7):
            _,img_rs_L,_,img_rs_ab = preprocess_img(load_img(os.path.join(self.data_path,'DAVIS', self.frame_list[i]).replace('\\', '/')))
            images.append(img_rs_L)
            if i == index + 7//2:
                img_rs_ab_center = img_rs_ab
        img_rs_L = torch.stack(images, 0)
        return img_rs_L, img_rs_ab_center


class VCdataset_test(Dataset):
    def __init__(self):
        self.data_path = 'sample dataset'
        self.frame_list = os.path.join(self.data_path, 'sample_davis_test.txt')
        self.file = open(self.frame_list, 'r')
        self.lines = self.file.read()
        self.frame_list = self.lines.split('\n')

    def __getitem__(self, index):
        images = []
        for i in range(index, index + 7):
            _,img_rs_L,_,_ = preprocess_img(load_img(os.path.join(self.data_path,'DAVIS', self.frame_list[i]).replace('\\', '/')))
            images.append(img_rs_L)
        img_rs_L = torch.stack(images, 0)
        return img_rs_L
