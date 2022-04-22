import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchsummary import summary
import matplotlib.pyplot as plt

from utils import *
from model import VCNet
from dataloader import *

if __name__ == "__main__":
    img = load_img(r"sample dataset\DAVIS-gray\car-turn\00017.jpg")
    img = resize_img(img)
    original_l, _ = preprocess_img(img)

    # model = ECCVGenerator()
    # model.load_state_dict(torch.load(r"colorization_weights.pth"))
    model = VCNet()
    model.eval()
    # out = model(original_l)

    # output_image = postprocess_tens(original_l, out)
    # print("here")
    # plt.imsave("car.png", output_image)
    summary(model, (1, 256, 256))
