# %%
import numpy as np
import cv2 as cv
from util import *


green_lower = np.array([30, 100, 0])
green_upper = np.array([200, 255, 255])


def segmentation(src, dest):
    image = cv2.imread(src)
    image = image[1100:2500, 1700:3400, :]
    
    origin = image.copy()

    image[image < 160] = 0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, green_lower, green_upper)
    hsv[mask == 0] = 0
    mask = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image -= mask

    image = cv2.GaussianBlur(image, (25, 25), cv2.BORDER_DEFAULT)
    image = cv2.GaussianBlur(image, (25, 25), cv2.BORDER_DEFAULT)
    image = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
    # https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv.kmeans(
        Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    res2[res2 > 20] = 255

    sss = origin.copy()
    sss.fill(0)
    sss[:, :, 0] = res2
    sss[:, :, 1] = res2
    sss[:, :, 2] = res2

    res3 = cv2.bitwise_and(origin, sss)

    save_img(origin, 'original.jpg')
    save_img(res3, dest)
