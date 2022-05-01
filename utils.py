# Utility and helper functions
import os
import yaml
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F

# from torchvision import transforms


def header(head):
    print("-" * 80)
    print(f"\t\t\t\t{head.upper()}")
    print("-" * 80)


def verify_config(config):
    assert config["Encoder"]["context"] == config["Setup"]["context"]


def generate_model_name(config):
    model_name = f"{config['model']}_{config['dataset']}_{config['batch_size']}_{config['lr']}_{config['epochs']}_{config['context']}"
    return model_name


def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if out_np.ndim == 2:
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np


def resize_img(img, HW=(256, 256), resample=3):
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))


def preprocess_img(img_rgb_orig, HW=(256, 256), resample=3):
    # return original size L and resized L as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    # img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)
    # img_ab_rs = np.zeros((64,64,2))

    # img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]
    # img_ab_orig = img_lab_orig[:, :, 1:]
    img_ab_rs = img_lab_rs[:, :, 1:]

    # img_ab_rs = np.asarray(Image.fromarray(transforms.Resize(64)(transforms.ToPILImage(img_lab_rs[:, :, 1:]))))
    # img_ab_rs[:, :, 0] =resize_img(img_lab_rs[:, :, 1], (64, 64))
    # img_ab_rs[:, :, 1] =resize_img(img_lab_rs[:, :, 2], (64, 64))

    tens_orig_l = None
    tens_orig_ab = None

    # tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]  # 1 x 1 x H_orig x W_orig
    tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :]  # 1 x 1 x H_rs x W_rs
    # tens_orig_ab = torch.Tensor(img_ab_orig)[None, :, :, :]  # 1 x 2 x H_orig x W_orig
    tens_rs_ab = torch.Tensor(img_ab_rs)[None, :, :, :]  # 1 x 2 x H_rs x W_rs

    return (tens_orig_l, tens_rs_l, tens_orig_ab, tens_rs_ab)


def postprocess_tens(tens_orig_l, out_ab, mode="bilinear"):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode="bilinear")
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))


l_cent = 50.0
l_norm = 100.0
ab_norm = 110.0


def normalize_l(in_l):
    return (in_l - l_cent) / l_norm


def unnormalize_l(in_l):
    return in_l * l_norm + l_cent


def normalize_ab(in_ab):
    return in_ab / ab_norm


def unnormalize_ab(in_ab):
    return in_ab * ab_norm
