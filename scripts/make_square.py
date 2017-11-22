#! /usr/bin/python
import tarfile

import cv2
import math
import numpy as np

# Still might be useful to find horizon to find the best way to crop
import shutil

from scripts.generate_splits import generate_vgg_splits
from scripts.util import make_tarfile, show


def resize_to_square(img):
    if img.shape[0] > 224:
        rat = 224 / img.shape[0]
        img = cv2.resize(img, (0, 0), fx=rat, fy=rat)
    if img.shape[1] > 224:
        mid = int(math.ceil(img.shape[1] / 2))
        img = img[:, (mid-112):(mid+112)]

    if img.shape[0] < 224:
        difference = int(math.ceil((224-img.shape[0]) / 2))
        img = cv2.copyMakeBorder(img,difference, difference,0,0, cv2.BORDER_CONSTANT)
        # raise Exception("Unfamiliar dimension")
    if img.shape[1] < 224:
        difference = int(math.ceil((224-img.shape[1]) / 2))
        img = cv2.copyMakeBorder(img,0,0,difference, difference, cv2.BORDER_CONSTANT)

    img = img[:224,:224,:]


    assert img.shape == (224,224,3)
    return img

def zoom(img):
    y,x,_ = img.shape
    chunk = int(.15*y)
    return img[chunk:y-chunk,:,:]

def reformat(path,target, img, folder, listfile):
    new_img = resize_to_square(img)
    if target is '0':
        new_path = './' + folder + '/shanghai/' + path.split('/')[-1]
        cv2.imwrite(new_path, new_img)
    elif target is '1':
        new_path = './' + folder + '/london/' + path.split('/')[-1]
        cv2.imwrite(new_path, new_img)
    elif target is '2':
        new_path = './' + folder + '/nyc/' + path.split('/')[-1]
        cv2.imwrite(new_path, new_img)
    elif target is '3':
        new_path = './' + folder + '/chicago/' + path.split('/')[-1]
        cv2.imwrite(new_path, new_img)
    else:
        raise Exception('bad target')
    listfile.write(new_path + ',' + target + '\n')

if __name__ == '__main__':
    import sys, os

    try:
        folder, listfile = sys.argv[1:3]
        os.mkdir(folder)
    except IndexError:
        folder, listfile = './tmp_data', './tmp_data/tmp_data.txt'
    except OSError:
        folder, listfile = sys.argv[1:3]

    for city in ['london','shanghai', 'nyc','chicago']:
        try:
            os.mkdir(folder + '/' + city)
        except OSError:
            pass

    new_list = open(listfile,'w+')
    for line in open('./lists/skyline_data.txt').readlines():
        path, target = line.strip('\n').split(',')

        img = cv2.imread(path, cv2.IMREAD_COLOR)

        # if len(img.shape) > 2:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = np.dstack([img, img, img])
        img = zoom(img)
        # show(resize_to_square(img))
        reformat(path, target, img, folder, new_list)

    generate_vgg_splits(folder=folder, listpath=listfile)

    make_tarfile(folder+'.tgz', folder)
