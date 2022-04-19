import os
import numpy as np

data_path = r"sample dataset"  # sample
# data_path = r"dataset\dataset\DAVIS_videvo_train"  # main dataset


davis_rgb = os.listdir(os.path.join(data_path, "DAVIS"))
davis_gray = os.listdir(os.path.join(data_path, "DAVIS-gray"))

# videvo_rgb = os.listdir(os.path.join(data_path, "videvo"))
# videvo_gray = os.listdir(os.path.join(data_path, "videvo-gray"))

with open("sample_davis_train.txt", "w") as dav:
    for video in davis_rgb:
        for f in os.listdir(os.path.join(data_path, "DAVIS", video)):
            dav.write(os.path.join(video, f) + "\n")

# with open("sample_videvo_train.txt", "w") as viv:
#     for video in videvo_rgb:
#         for f in os.listdir(os.path.join(data_path, "videvo", video)):
#             viv.write(os.path.join(video, f) + "\n")

# with open("both_train.txt", "w") as both:
#     combined = []
#     for video in davis_rgb:
#         for f in os.listdir(os.path.join(data_path, "DAVIS", video)):
#             combined.append(os.path.join(video, f))

#     for video in videvo_rgb:
#         for f in os.listdir(os.path.join(data_path, "videvo", video)):
#             combined.append(os.path.join(video, f))

#     combined.sort()
#     for i in combined:
#         both.write(i + "\n")
