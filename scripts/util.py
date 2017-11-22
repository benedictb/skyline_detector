import os
import random
import tarfile

import cv2
import numpy as np

label_dict = {0:'Shanghai',1:'London',2:'New York City', 3:'Chicago'}

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def data_loader(filepath='/lists/skyline_data.txt', channels=cv2.IMREAD_GRAYSCALE, randomize=False):
    data = [ tuple(l.strip('\n').split(',')) for l in open(filepath) ]
    if randomize:
        random.shuffle(data)

    for d in data:
        path, label = d
        # data.pop(0)
        # print(path)
        img = cv2.imread(path, channels)
        yield (img, label)

def condense(l):
    m = l[0]
    for arr in l[1:]:
        m = np.append(m, arr, axis=0)
    return m


def counter(n=0):
    while True:
        yield n
        n += 1

def accuracy(l):
    return sum([1 if t[0] == t[1] else 0 for t in l]) / float(len(l))

def show(img, title='image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()