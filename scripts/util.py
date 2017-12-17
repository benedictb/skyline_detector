import os
import random
import tarfile

import cv2
import numpy as np
import sklearn.metrics

label_dict = {'0': 'Shanghai', '1': 'London', '2': 'New York City', '3': 'Chicago'}
label_list = ['Shanghai', 'London', 'New York City', 'Chicago']


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def data_loader(filepath='/lists/skyline_data.txt', channels=cv2.IMREAD_GRAYSCALE, randomize=False, preprocess=None):
    data = [tuple(l.strip('\n').split(',')) for l in open(filepath)]
    if randomize:
        random.shuffle(data)

    for d in data:
        try:
            path, label = d
            img = cv2.imread(path, channels)
            for func in preprocess:
                img = func(img)
        except Exception as e:
            print(path)
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


def chiSquared(p, q):
    return 0.5 * np.sum((p - q) ** 2 / (p + q + 1e-6))


def accuracy(l):
    return sum([1 if t[0] == t[1] else 0 for t in l]) / float(len(l))


def show(img, title='image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def res_print(res):
    for pred, target in res:
        print('Pred:{} Target:{}'.format(pred[0], target))

    print('\n\nACC:{}'.format(accuracy(res)))


def fscore(res):
    truth = np.asarray([l[1] for l in res])
    pred = np.asarray([l[0][0] for l in res])
    return sklearn.metrics.f1_score(truth, pred, average='weighted')


def fscore2(truth, pred):
    return sklearn.metrics.f1_score(truth, pred, average='weighted')


def class_report(res):
    truth = np.asarray([int(l[1]) for l in res])
    pred = np.asarray([int(l[0][0]) for l in res])
    return sklearn.metrics.classification_report(truth, pred, target_names=label_list)


def class_report2(truth, pred):
    return sklearn.metrics.classification_report(truth, pred, target_names=label_list)
