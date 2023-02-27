# %%
import numpy as np
import pandas as pd
import cv2
from torch import bitwise_not


def save_img(image, name=None):
    cv2.imwrite(name if name else 'image_balloon.jpg', image)


def preprocess(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.mean(hsv)[1]
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    save_img(image, "hsv.jpg")

    image = cv2.bitwise_not(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    image = clahe.apply(image)

    save_img(image, "clahe.jpg")

    image = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
    # image = cv2.Canny(image, 30, 150, 3)
    # image = cv2.dilate(image, (1, 1), iterations=0)
    return image


def contouring(image, objType=None):
    im = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)

    contours, hierarchy = cv2.findContours(
        im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(im, contours, -1, (0, 255, 0), 7
                           )
    return img, contours


def load_dataset(path):
    bbox = pd.read_csv(path)
    bbox["bbox"] = [eval(x[1:-1]) for x in bbox["bbox"]]
    return bbox


def crop_bbox(target):
    image = cv2.imread(path+target[0])
    image = image[
        target["bbox"]["ymin"]:target["bbox"]["ymax"],
        target["bbox"]["xmin"]:target["bbox"]["xmax"], :]
    return image
