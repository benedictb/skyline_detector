import cv2
import math

import pickle
import scipy.stats
import sklearn
import tqdm

from util import data_loader, show
from preprocess import grayscale, resize_to_square, zoomy, chop_lower, resize

from util import fscore2, class_report2

PREPROCESS_QUEUE = [grayscale, resize]
FEATURES = 100
RANDOM = False
CHANNELS = cv2.IMREAD_COLOR
FLANN_INDEX_KDTREE = 0
USE_N_MATCHES = 30
label_dict = {'0': 'Shanghai', '1': 'London', '2': 'New York City', '3': 'Chicago'}


class BF(object):
    def __init__(self, trainpath, testpath, n=3, detector='orb', use_n_matches=30):
        self.trainstream = data_loader(filepath=trainpath, channels=CHANNELS,
                                       preprocess=PREPROCESS_QUEUE, randomize=RANDOM)
        self.teststream = data_loader(filepath=testpath, channels=CHANNELS,
                                      preprocess=PREPROCESS_QUEUE, randomize=RANDOM)

        self.num_matches = use_n_matches

        if detector.lower() == 'orb':
            self.KPD = cv2.ORB_create(nfeatures=FEATURES, scoreType=cv2.ORB_FAST_SCORE)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif detector.lower() == 'surf':
            self.KPD = cv2.xfeatures2d.SURF_create(hessianThreshold=1000)
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            self.bf = cv2.FlannBasedMatcher(index_params, search_params)
        elif detector.lower() == 'sift':
            self.KPD = cv2.xfeatures2d.SIFT_create(nfeatures=FEATURES)
            self.bf = cv2.BFMatcher(crossCheck=True)
        else:
            print('Unknown Keypoint Detector')
            exit(1)

        # Nearest neighbors
        self.n = n

        # Descriptor dictionary
        self.desc = dict()

    def train(self):
        print('Training...')
        labels = []
        for i, v in tqdm.tqdm(enumerate(self.trainstream), total=599):
            data = v[0]
            target = v[1]
            _, des = self.KPD.detectAndCompute(data, None)
            self.desc[i] = des
            labels.append(target)
        self.labels = labels

    def test(self):
        print('Testing...')
        preds = []
        test_labels = []
        for img, target in tqdm.tqdm(self.teststream, total=150):
            test_labels.append(target)
            _, des = self.KPD.detectAndCompute(img, None)
            res = []
            for index, tdes in self.desc.items():

                matches = self.bf.match(des, tdes)
                matches = sorted(matches, key=lambda x: x.distance)
                try:
                    distance = sum([1 / float(x.distance) for x in matches[:self.num_matches]]) / float(
                        max(self.num_matches, len(matches)))
                except ZeroDivisionError:
                    distance = 1
                res.append((index, distance))

            res = sorted(res, key=lambda x: x[1], reverse=True)
            top = [self.labels[l[0]] for l in res[:self.n]]

            if self.n == 0:
                preds.append(top[0])
            else:
                items, count = scipy.stats.mode(top)
                preds.append(items[0][max(range(len(items)), key=lambda x: count[x])])

        return preds, test_labels

    # trying things new here don't know why
    def test_single(self):
        saved = pickle.load(open("model.p", "rb"))
        t_labels = saved['labels']
        t_desc = saved['desc']

        img, target = next(self.teststream)

        _, des = self.KPD.detectAndCompute(img, None)
        res = []
        for index, tdes in t_desc.items():
            matches = self.bf.match(des, tdes)
            matches = sorted(matches, key=lambda x: x.distance)
            try:
                distance = sum([1 / float(x.distance) for x in matches[:self.num_matches]]) / float(
                    max(self.num_matches, len(matches)))
            except ZeroDivisionError:
                distance = 1
            res.append((index, distance))

        res = sorted(res, key=lambda x: x[1], reverse=True)
        top = [t_labels[l[0]] for l in res[:self.n]]

        if self.n == 0:
            pred = top[0]
        else:
            items, count = scipy.stats.mode(top)
            pred = items[0][max(range(len(items)), key=lambda x: count[x])]

        print('Label: {} ({})'.format(target, label_dict[target]))
        print('Prediction: {} ({})'.format(pred, label_dict[pred]))
        show(img)


if __name__ == '__main__':
    import random

    # split = random.randint(0, 9)
    # bf = BF('./lists/splits/{}_train.txt'.format(split),
    #         './lists/splits/{}_test.txt'.format(split),
    #         n=1,
    #         detector='surf',
    #         use_n_matches=NUM_MATCHES)

    bf = BF('./lists/skyline_data.txt',
            './lists/skyline_test_data.txt',
            n=1,
            detector='orb',
            use_n_matches=USE_N_MATCHES)

    bf.train()
    preds, labels = bf.test()
    print(fscore2(labels, preds))
    print(class_report2(labels, preds))
    print(sklearn.metrics.accuracy_score(labels, preds))
