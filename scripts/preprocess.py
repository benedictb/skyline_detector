import statistics

import cv2
import math
import numpy as np

from util import data_loader, show


def blur(img, n=5):
    return cv2.medianBlur(img, n)


def sharpen(img):
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 4.0  # Identity, times two!

    # Create a box filter:
    boxFilter = np.ones((9, 9), np.float32) / 81.0
    # Subtract the two:
    kernel = kernel - boxFilter

    return cv2.filter2D(img, -1, kernel)


def binarize(img):
    img = blur(img, n=5)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img

def inverse(img):
     return cv2.bitwise_not(img)


def erode(img):
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    mask = mask / 255
    out = img * mask
    return out

# BG SUb
def reverse_bg_subtraction(img):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    return fgbg.apply(img)


def find_horizon(img):
    # Apply edge detection method on the image
    edges = cv2.Canny(img, 50, 150, apertureSize=5)

    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    for l in lines:
        r, theta = l[0]
        if theta < 1.588 and theta > 1.55:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2)

    return img


def resize(img):
    prod = img.shape[0] * img.shape[0]
    if prod > 750000:
        rat = 750000 / prod
        img = cv2.resize(img, (0, 0), fx=rat, fy=rat)
    return img


if __name__ == '__main__':
    loader = data_loader(filepath='/home/benedict/classes/cv/project/lists/splits/0_train.txt',
                         channels=cv2.IMREAD_GRAYSCALE, randomize=False)
    # loader = data_loader(filepath='/home/benedict/classes/cv/project/lists/splits/0_train.txt', channels=cv2.IMREAD_COLOR, randomize=True)

    for img, target in loader:
        img = resize(img)
        img = binarize(img)
        img = inverse(img)
        # img = erode(img)
        # img = find_horizwon(img)
        # img = reverse_bg_subtraction(img)
        show(img)
