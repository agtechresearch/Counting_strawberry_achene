# %%
import numpy as np
import pandas as pd
import cv2
import os
from util import *

import logging
logging.basicConfig(filename='achene.csv', filemode='w',
                    format='%(message)s', level=logging.INFO)
LOG = logging.getLogger(__name__)
lower = np.array([22, 93, 30])
upper = np.array([45, 255, 255])

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10
params.maxThreshold = 500
params.filterByArea = True
params.minArea = 30
params.maxArea = 2500
params.filterByCircularity = True
params.minCircularity = 0
params.filterByConvexity = True
params.minConvexity = 0
params.filterByInertia = True
params.minInertiaRatio = 0


def achene(src, dest_blob, dest_diff):
    image = cv2.imread(src)
    h, w, c = image.shape
    origin = image.copy()
    area = origin[origin > 5].size
    image[image > 240] = 220

    image = preprocess(image)
    image = cv2.GaussianBlur(image, (25, 25), cv2.BORDER_DEFAULT)
    image = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)

    image, contours = contouring(image)
    image = cv2.Canny(image, 50, 100)

    image = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
    image[image == 0] = 255
    image[image < 255] = 0

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(image)
    points = ""
    for k in keypoints:
        points += str(k.pt[0]/w) + "," + str(k.pt[1]/h) + ","

    image = cv2.drawKeypoints(image, keypoints, None, (0, 0, 255),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    
    LOG.info(f"{os.path.basename(src)},{len(keypoints)},{area}, {points}")

    image = cv2.putText(image, f"Total : {str(len(keypoints))}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)

    save_img(image, dest_blob)
    save_img(origin, "origin.jpg")
    save_img(image, "blob.jpg")

    img3 = cv2.absdiff(image, origin)

    save_img(img3, dest_diff)
    save_img(img3, "diff.jpg")
    image = cv2.erode(image, None, iterations=5)
    h, w, _ = image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
