import os
from matplotlib import lines
from utils import *
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class VCSamples(Dataset):
    def __init__(self, config):
        self.data_path = config["dataset_path"]
        self.frame_list = os.path.join(self.data_path, config["dataset_list"] + "_train.txt")
        self.context = config["Encoder"]["context"]
        self.n_frames = self.context * 2 + 1

        with open(self.frame_list, "r") as f:
            self.lines = f.read()
        self.frame_list = self.lines.split("\n")

    def __len__(self):
        return len(self.frame_list) - 2 * self.context

    def __getitem__(self, index):
        images = []
        for i in range(index, index + self.n_frames):
            _, img_rs_L, _, img_rs_ab = preprocess_img(load_img(os.path.join(self.data_path, self.frame_list[i]).replace("\\", "/")))
            images.append(img_rs_L)
            if i == index + self.n_frames // 2:
                img_rs_ab_center = img_rs_ab
        img_rs_L = torch.stack(images, 0)
        img_rs_L = img_rs_L.squeeze()
        img_rs_ab_center = img_rs_ab_center.squeeze()

        # print("Data", img_rs_L.shape, img_rs_ab_center.shape)
        img_rs_ab_center = torch.transpose(img_rs_ab_center, 0, 2)
        img_rs_ab_center = torch.transpose(img_rs_ab_center, 1, 2)
        # print("After Transpose", img_rs_L.shape, img_rs_ab_center.shape)
        return img_rs_L, img_rs_ab_center


class VCSamples_Test(Dataset):
    def __init__(self, config):
        self.data_path = config["dataset_path"]
        # self.frame_list = os.path.join(data_path, "sample_davis_train.txt")
        self.frame_list = os.path.join(self.data_path, config["dataset_list"] + "_train.txt")
        with open(self.frame_list, "r") as f:
            self.lines = f.read()

        self.frame_list = self.lines.split("\n")
        self.context = config["Encoder"]["context"]
        self.n_frames = self.context * 2 + 1

    def __len__(self):
        return len(self.frame_list) - 2 * self.context

    def __getitem__(self, index):
        images = []
        for i in range(index, index + self.n_frames):
            _, img_rs_L, _, _ = preprocess_img(load_img(os.path.join(self.data_path, self.frame_list[i]).replace("\\", "/")))
            # img_rs_L = torch.tensor(plt.imread(os.path.join(self.data_path,'DAVIS', self.frame_list[i]).replace('\\', '/')))
            images.append(img_rs_L)
        img_rs_L = torch.stack(images, 0)
        return img_rs_L.squeeze().unsqueeze(0)


if __name__ == "__main__":
    with open("attention_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    test_dataset = VCSamples_Test(config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )
    print(len(test_loader))
    for i, img in enumerate(test_loader):
        print(img.shape)
        plt.imshow(img[0][0].numpy())
        plt.show()
        if i == 1:
            break
