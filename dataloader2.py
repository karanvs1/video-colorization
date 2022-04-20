import os
from matplotlib import lines
from utils import *
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class VCSamples(Dataset):
    def __init__(self, context):
        self.data_path = 'sample dataset'
        self.frame_list = os.path.join(self.data_path, 'sample_davis_train.txt')
        self.file = open(self.frame_list, 'r')
        self.lines = self.file.read()
        self.frame_list = self.lines.split('\n')
        self.context = context
        self.n_frames = self.context * 2 + 1

    def __len__(self):
        return len(self.frame_list) - 2 * self.context

    def __getitem__(self, index):
        images = []
        for i in range(index, index + self.n_frames):
            _,img_rs_L,_,img_rs_ab = preprocess_img(load_img(os.path.join(self.data_path,'DAVIS', self.frame_list[i]).replace('\\', '/')))
            images.append(img_rs_L)
            if i == index + self.n_frames//2:
                img_rs_ab_center = img_rs_ab
        img_rs_L = torch.stack(images, 0)
        return img_rs_L.squeeze(), img_rs_ab_center.squeeze()


class VCSamples_Test(Dataset):
    def __init__(self, context):
        self.data_path = 'sample dataset'
        self.frame_list = os.path.join(self.data_path, 'sample_davis_train.txt')
        self.file = open(self.frame_list, 'r')
        self.lines = self.file.read()
        self.frame_list = self.lines.split('\n')
        self.context = context
        self.n_frames = self.context * 2 + 1

    def __len__(self):
        return len(self.frame_list) - 2 * self.context

    def __getitem__(self, index):
        images = []
        for i in range(index, index + self.n_frames):
            _,img_rs_L,_,_ = preprocess_img(load_img(os.path.join(self.data_path,'DAVIS', self.frame_list[i]).replace('\\', '/')))
            # img_rs_L = torch.tensor(plt.imread(os.path.join(self.data_path,'DAVIS', self.frame_list[i]).replace('\\', '/')))
            images.append(img_rs_L)
        img_rs_L = torch.stack(images, 0)
        return img_rs_L.squeeze()


if __name__ == '__main__':
    with open('test_config.yaml', "r") as f:
        config = yaml.safe_load(f)
    test_dataset = VCSamples_Test(config["PreprocessNet"]["context"])
    test_loader = torch.utils.data.DataLoader(test_dataset, **config["Setup"]["train_dataloader"])
    print(len(test_loader))
    for i, img in enumerate(test_loader):
        print(img.shape)
        plt.imshow(img[0][0].numpy())
        plt.show()
        if i == 1:
            break