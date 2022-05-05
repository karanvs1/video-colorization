import os
import numpy as np

data_path = r"sample_dataset"  # sample
# data_path = r"actual_dataset"  # main dataset
# data_path = r"video-test"  # test dataset

davis_rgb = os.listdir(os.path.join(data_path, "DAVIS"))
# davis_gray = os.listdir(os.path.join(data_path, "DAVIS-gray"))

# videvo_rgb = os.listdir(os.path.join(data_path, "videvo"))
# videvo_gray = os.listdir(os.path.join(data_path, "videvo-gray"))

with open(os.path.join(data_path, "sample_davis_train.txt"), "w") as dav:
    for video in davis_rgb:
        for f in sorted(os.listdir(os.path.join(data_path, "DAVIS", video))):
            dav.write("\n" + os.path.join('DAVIS', video, f))



########################################################################################################################
# #video test
# frame_list = sorted(list(map (lambda x: int(x) , list(map(lambda x: x.split('.')[0], os.listdir(os.path.join(data_path)))))))
# # print(frame_list)
# with open(os.path.join(data_path, "video_test_train.txt"), "w") as dav:
#     # for video in davis_rgb:
#     for f in frame_list:
#         dav.write(str(f)+".jpg" + "\n")
######################################################################################################################## 
    
    
    # print(frame_list)
    # for f in sorted(os.listdir(os.path.join(data_path))):
    #     dav.write( "\n" + os.path.join('video-test', f))


# with open(os.path.join(data_path, "sample_videvo_train.txt"), "w") as viv:
#     for video in videvo_rgb:
#         for f in os.listdir(os.path.join(data_path, "videvo", video)):
#             viv.write(os.path.join('videvo', video, f) + "\n")

# with open(os.path.join(data_path, "both_train.txt"), "w") as both:
#     combined = []
#     for video in davis_rgb:
#         for df in os.listdir(os.path.join(data_path, "DAVIS", video)):
#             combined.append(os.path.join('DAVIS', video, df))

#     for video in videvo_rgb:
#         for vf in os.listdir(os.path.join(data_path, "videvo", video)):
#             combined.append(os.path.join('videvo', video, vf))

#     for i in combined:
#         both.write("\n" + i)
