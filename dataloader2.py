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

    def __len__(self):
        return len(self.frame_list) - 6

    def __getitem__(self, index):
        images = []
        for i in range(index, index + 7):
            _,img_rs_L,_,img_rs_ab = preprocess_img(load_img(os.path.join(self.data_path,'DAVIS', self.frame_list[i]).replace('\\', '/')))
            images.append(img_rs_L)
            if i == index + 7//2:
                img_rs_ab_center = img_rs_ab
        img_rs_L = torch.stack(images, 0)
        return img_rs_L.squeeze(), img_rs_ab_center.squeeze()


class VCdataset_test(Dataset):
    def __init__(self):
        self.data_path = 'sample dataset'
        self.frame_list = os.path.join(self.data_path, 'sample_davis_train.txt')
        self.file = open(self.frame_list, 'r')
        self.lines = self.file.read()
        self.frame_list = self.lines.split('\n')

    def __len__(self):
        return len(self.frame_list) - 6

    def __getitem__(self, index):
        images = []
        for i in range(index, index + 7):
            _,img_rs_L,_,_ = preprocess_img(load_img(os.path.join(self.data_path,'DAVIS', self.frame_list[i]).replace('\\', '/')))
            # img_rs_L = torch.tensor(plt.imread(os.path.join(self.data_path,'DAVIS', self.frame_list[i]).replace('\\', '/')))
            images.append(img_rs_L)
        img_rs_L = torch.stack(images, 0)
        return img_rs_L.squeeze()


test_loader = torch.utils.data.DataLoader(VCdataset_test(), batch_size=1, shuffle=False, num_workers=0)
for i, img in enumerate(test_loader):
    print(img.shape)
    plt.imshow(img[0][0].numpy())
    plt.show()
    if i == 1:
        break