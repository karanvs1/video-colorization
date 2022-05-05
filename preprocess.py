import os
import skimage
from utils import preprocess_img, load_img
from tqdm import tqdm
import matplotlib.pyplot as plt

data_path = r".\sample_dataset"
new_data_path = r".\sample_dataset\DAVIS-light"
frame_list_path = os.path.join(r".\sample_dataset", "sample_davis_train.txt")
try:
    os.mkdir(new_data_path)
except Exception as e:
    print(e)

with open(frame_list_path, "r") as f:
    lines = f.read()
frame_list = lines.split("\n")

for i in tqdm(range(len(frame_list))):
    _, img_rs_L, _, _ = preprocess_img(load_img(os.path.join(data_path, frame_list[i]).replace("\\", "/")))
    save_path = os.path.join(new_data_path, frame_list[i].split("/")[-1])
    # print(img_rs_L.shape)
    plt.imsave(save_path, img_rs_L)
