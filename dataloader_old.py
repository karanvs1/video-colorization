# DAVIS and Videvo Dataloader
import cv2 as cv
from torch.utils.data import Dataset


class VCSamples(Dataset):
    def __init__(self, config):
        self.dataset_path = config["dataset_path"]
        self.frame_list = os.path.join(self.dataset_path, config["dataset_list"] + "_train.txt")

    def __getitem__(self, index):
        imgpath = self.frame_list[index]  # Path of one image
        # Read the images
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB image

        grayimg = img[:, :, [0]] * 0.299 + img[:, :, [1]] * 0.587 + img[:, :, [2]] * 0.114
        grayimg = np.concatenate((grayimg, grayimg, grayimg), axis=2)
        # Data augmentation
        grayimg = self.img_aug(grayimg)
        img = self.img_aug(img)
        # Normalized to [-1, 1]
        grayimg = np.ascontiguousarray(grayimg, dtype=np.float32)
        grayimg = (grayimg - 128) / 128
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = (img - 128) / 128
        # To PyTorch Tensor
        grayimg = torch.from_numpy(grayimg).permute(2, 0, 1).contiguous()
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return grayimg, img

    def __len__(self):
        return len(self.frame_list)

    def collate_fn(self, batch):
        pass


class ColorizationDataset(Dataset):
    def __init__(self, opt):
        # Note that:
        # 1. opt: all the options
        # 2. imglist: all the image names under "baseroot"
        self.opt = opt
        self.imglist = self.get_files(opt.baseroot)

    def get_files(self, path):
        # Read a folder, return the complete path
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))
        # Randomly sample the target slice
        sample_size = int(math.floor(len(ret) / self.opt.sample_size))
        ret = random.sample(ret, sample_size)
        # Re-arrange the list that meets multiplier of batchsize
        adaptive_len = int(math.floor(len(ret) / self.opt.batch_size) * self.opt.batch_size)
        ret = ret[:adaptive_len]
        return ret

    def img_aug(self, img):
        # Random scale
        """
        if self.opt.geometry_aug:
            H_in = img.shape[0]
            W_in = img.shape[1]
            sc = np.random.uniform(self.opt.scale_min, self.opt.scale_max)
            H_out = int(math.floor(H_in * sc))
            W_out = int(math.floor(W_in * sc))
            # scaled size should be greater than opts.crop_size and remain the ratio of H to W
            if H_out < W_out:
                if H_out < self.opt.crop_size:
                    H_out = self.opt.crop_size
                    W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
            else: # W_out < H_out
                if W_out < self.opt.crop_size:
                    W_out = self.opt.crop_size
                    H_out = int(math.floor(H_in * float(W_out) / float(W_in)))
            img = cv2.resize(img, (W_out, H_out))
        """
        if self.opt.geometry_aug:
            H_in = img.shape[0]
            W_in = img.shape[1]
            if H_in < W_in:
                H_out = self.opt.crop_size
                W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
            else:  # W_in < H_in
                W_out = self.opt.crop_size
                H_out = int(math.floor(H_in * float(W_out) / float(W_in)))
            img = cv2.resize(img, (W_out, H_out))
        else:
            img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))
        # Random crop
        cropper = RandomCrop(img.shape[:2], (self.opt.crop_size, self.opt.crop_size))
        img = cropper(img)
        """
        # Random rotate
        if self.opt.angle_aug:
            # Rotate
            rotate = random.randint(0, 3)
            if rotate != 0:
                img = np.rot90(img, rotate)
            # Horizontal flip
            if np.random.random() >= 0.5:
                img = cv2.flip(img, flipCode = 1)
        """
        return img
