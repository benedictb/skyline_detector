import random

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from preprocess import resize
from util import data_loader, condense, counter, accuracy
from sklearn.cluster import KMeans, MiniBatchKMeans

from preprocess import grayscale, resize_to_square, zoomy, chop_lower

# from scripts.preprocess import chop_lower
from clf import SVM, MLP, KNN, SVM_chi2
from util import show, fscore, class_report, fscore2, class_report2

CHANNELS = cv2.IMREAD_COLOR




# PARAMS (SVM)
PREPROCESS_QUEUE = [chop_lower, grayscale, resize]
VOCAB_SIZE = 100
RANDOM = True
FEATURES = 500


# FEATURES= 3000


class BOV(object):
    def __init__(self, trainpath, testpath, detector='sift'):
        self.trainstream = data_loader(filepath=trainpath, channels=CHANNELS,
                                       preprocess=PREPROCESS_QUEUE, randomize=RANDOM)
        self.teststream = data_loader(filepath=testpath, channels=CHANNELS,
                                      preprocess=PREPROCESS_QUEUE, randomize=RANDOM)
        # self.SIFT = cv2.xfeatures.SIFT_create()

        # Model
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

        # Cluster method
        self.kMeans = MiniBatchKMeans(VOCAB_SIZE, batch_size=64)

    def build_hist(self, features, cluster=False):
        m = condense(features)
        if cluster:
            print("Making vocab...")
            print('Clustering...')
            preds = self.kMeans.fit_predict(m)
        else:
            preds = self.kMeans.predict(m)
        length = len(features)
        hist = np.zeros([length, VOCAB_SIZE])
        c = counter()
        print('Making histogram...')
        for i in range(length):
            for _ in range(len(features[i])):
                word = preds[next(c)]
                hist[i][word] += 1
        if cluster:
            print('Vocab complete')
        return hist
        # self.hist = hist

    def get_features(self, stream):
        print('Gathering features...')
        descriptors = []
        labels = []
        for img, target in stream:
            kp, des = self.KPD.detectAndCompute(img, None)
            # kp, des = self.KPD.detect(img, None)
            # _, des = self.KPD.compute(img, kp)
            descriptors.append(des)
            labels.append(target)
        print('Features gathered')
        return descriptors, np.asarray(labels)

    def get_test_data(self):
        data, labels = self.get_features(self.teststream)
        hist = self.build_hist(data)
        hist = self.samplewise_min_max(hist)
        hist = self.scale.transform(hist)
        return hist, np.asarray(labels)

    def get_train_data(self):
        data, labels = self.get_features(self.trainstream)

        self.hist = self.build_hist(data, cluster=True)
        self.hist = self.samplewise_min_max(self.hist)
        self.scale = self.standardize(self.hist)
        self.hist = self.scale.transform(self.hist)
        return self.hist, np.asarray(labels)

    @staticmethod
    def standardize(hist):
        return StandardScaler().fit(hist)

    @staticmethod
    def samplewise_min_max(hist):
        s = hist.sum(axis=1)
        return hist / s[:, None]

    def view_pics(self):
        for img, _ in self.trainstream:
            img = self.test_ORB(img)
            show(img)

    def test_ORB(self, img):
        kp = self.KPD.detect(img)
        img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), \
                                 flags=cv2.DrawMatchesFlags_DEFAULT)
        return img2


if __name__ == '__main__':

    # k = cv2.ORB_create(nfeatures=FEATURES, scoreType=cv2.ORB_FAST_SCORE)
    # for i in data_loader('./lists/skyline_test_data.txt', preprocess=PREPROCESS_QUEUE):
    #     kp = k.detect(i)
    #     img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), \
    #                              flags=cv2.DrawMatchesFlags_DEFAULT)
    #     show(img2)



    bov = BOV('./lists/skyline_test_data.txt',
              './lists/splits/0_test.txt',
              detector='orb')

    bov.view_pics()

    #
    '''SINGLE ITER'''
    # split = random.randint(0, 9)
    # print('FEATURES: {}\nVOCAB: {}\nSPLIT: {}'.format(FEATURES, VOCAB_SIZE, split))
    #
    # bov = BOV('./lists/splits/{}_train.txt'.format(split),
    #           './lists/splits/{}_test.txt'.format(split),
    #           detector='sift')
    #
    # # clf = SVM('poly')
    # clf = MLP(hidden_layers=(64, 16))
    # # clf = KNN(7)
    #
    # clf.train(*bov.get_train_data())
    # pred, labels = clf.test(*bov.get_test_data())
    # print(fscore2(labels, pred))
    # print(class_report2(labels, pred))

    '''THE WHOLE KIT & KABOODLE'''
    # print('FEATURES: {}\nVOCAB: {}\n'.format(FEATURES, VOCAB_SIZE))
    # scores = 0
    # for split in range(0,10):
    #     bov = BOV('./lists/splits/{}_train.txt'.format(split), './lists/splits/{}_test.txt'.format(split))
    #     clf = SVM('poly')
    #     # clf = KNN(9)
    #     # clf = MLP(hidden_layers=(128,64,8))
    #     clf.train(*bov.get_train_data())
    #     pred, labels = clf.test(*bov.get_test_data())
    #     print(class_report2(labels,pred))
    #     scores+=fscore2(labels, pred)
    #
    # print('AVE FSCORE:{}'.format(scores/float(10)))
    #
    # print('FEATURES: {}\nVOCAB: {}\n'.format(FEATURES, VOCAB_SIZE))
    # scores = 0
    # for split in range(0,10):
    #     bov = BOV('./lists/splits/{}_train.txt'.format(split), './lists/splits/{}_test.txt'.format(split))
    #     clf = SVM('poly')
    #     # clf = KNN(9)
    #     # clf = MLP(hidden_layers=(128,64,8))
    #     clf.train(*bov.get_train_data())
    #     pred, labels = clf.test(*bov.get_test_data())
    #     print(class_report2(labels,pred))
    #     scores+=fscore2(labels, pred)
    #
    # print('AVE FSCORE:{}'.format(scores/float(10)))

