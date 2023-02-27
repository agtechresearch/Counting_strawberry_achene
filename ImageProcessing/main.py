# %%
import numpy as np
import pandas as pd
import cv2

from util import *
from achene import *
from cluster import *
import time
import os

# for folder in os.listdir("./dataset/original"):
#     os.makedirs("./dataset/done/diff/" + folder, exist_ok=True)
#     os.makedirs("./dataset/segment/" + folder, exist_ok=True)
#     os.makedirs("./dataset/done/blob/" + folder, exist_ok=True)
#     os.makedirs("./dataset/done/blob/" + folder, exist_ok=True)

#     for file in os.listdir("./dataset/original/" + folder):
#         src = "./dataset/original/" + folder + "/" + file
#         # dest_seg = "./dataset/segment/" + folder + "/" + file
#         dest_blob = "./dataset/done/blob/" + folder + "/" + file
#         dest_diff = "./dataset/done/diff/" + folder + "/" + file

#         # segmentation(src, dest_seg)
#         achene(src, dest_blob, dest_diff)
#         time.sleep(2)


os.makedirs("./dataset/done/diff/all", exist_ok=True)
os.makedirs("./dataset/done/blob/all", exist_ok=True)

for file in os.listdir("./dataset/segment/all"):
    src = "./dataset/segment/all/" + file
    # dest_seg = "./dataset/segment/" + folder + "/" + file
    dest_blob = "./dataset/done/blob/" + file
    dest_diff = "./dataset/done/diff/" + file

    achene(src, dest_blob, dest_diff)
