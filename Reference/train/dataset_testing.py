#
# ! Sample Script for testing datasets

import argparse
import os
import time
import datetime
import os
import numpy as np
import cv2 as cv
import dataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_class_txt",
        type=str,
        default="./txt/DAVIS_videvo_train_class.txt",
        help="the path that contains DAVIS_videvo_train_class.txt",
    )
    parser.add_argument(
        "--video_imagelist_txt",
        type=str,
        default="./txt/DAVIS_videvo_train_imagelist.txt",
        help="the path that contains DAVIS_videvo_train_imagelist.txt",
    )
    # dataset
    parser.add_argument(
        "--baseroot",
        type=str,
        default="/mnt/d/CMU/Academics/Intro to DL/DL Project/video-colorization/Reference/dataset",
        help="color image baseroot",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    # ./ILSVRC2012_train
    # ./DAVIS_videvo_train
    parser.add_argument("--iter_frames", type=int, default=5, help="number of iter_frames in one iteration")
    parser.add_argument("--sample_size", type=int, default=1, help="sample number for the dataset at first stage")
    parser.add_argument("--crop_size", type=int, default=256, help="single patch size")  # first stage: 256 * 256
    parser.add_argument("--crop_size_h", type=int, default=256, help="single patch size")  # second stage (128p, 256p, 448p): 128, 256, 448
    parser.add_argument("--crop_size_w", type=int, default=448, help="single patch size")  # second stage (128p, 256p, 448p): 256, 448, 832
    parser.add_argument("--geometry_aug", type=bool, default=False, help="geometry augmentation (scaling)")
    parser.add_argument("--angle_aug", type=bool, default=False, help="geometry augmentation (rotation, flipping)")
    parser.add_argument("--scale_min", type=float, default=1, help="min scaling factor")
    parser.add_argument("--scale_max", type=float, default=1, help="max scaling factor")
    opt = parser.parse_args()
    print(opt)

    # Define the dataset
    trainset = dataset.ColorizationDataset(opt)
    print("The overall number of images:", len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)

    dataloader_iterator = iter(dataloader)
    gray, rgb = next(dataloader_iterator)

    print(type(gray), gray.shape)
    print(type(rgb), rgb.shape)

    cv.imshow("gray", gray[0].numpy())
    cv.imshow("rgb", rgb[0].numpy())
    cv.waitKey(1)
