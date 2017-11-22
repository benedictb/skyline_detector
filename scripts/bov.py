import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess import resize
from util import data_loader, condense, counter, accuracy
from sklearn.cluster import KMeans

### do this
PREPROCESS_QUEUE = [resize]
CHANNELS = cv2.IMREAD_GRAYSCALE
VOCAB_SIZE = 20

class BOV(object):
    def __init__(self, trainpath, testpath):
        self.trainstream = data_loader(filepath=trainpath, channels=CHANNELS)
        self.teststream = data_loader(filepath=testpath, channels=CHANNELS)
        # self.SIFT = cv2.xfeatures.SIFT_create()

        # Model
        self.ORB = cv2.ORB_create()

        # Cluster method
        self.kMeans = KMeans(VOCAB_SIZE)

        # Classifier
        self.svm = SVC()

    def train(self):
        features, labels = self.get_features(self.trainstream)
        self.make_vocab(features)
        self.standardize()
        self.svm.fit(self.hist, labels)

    def test(self):
        res = []
        for img, label in self.teststream:
            # In pred, label form
            res.append((self.predict(img), label))
        return res

    def make_vocab(self, features):
        m = condense(features, dim=3)
        print(m[0])
        print(len(m[0]))
        print(len(m) * VOCAB_SIZE)
        exit(0)
        preds = self.kMeans.fit_predict(m)
        length = len(features)
        hist = np.zeros([length, VOCAB_SIZE])
        c = counter()

        for i in range(length):
            for _ in range(len(features[i])):
                word = preds[next(c)]
                hist[i][word] += 1

        self.hist = hist

    def get_features(self,stream):
        descriptors = []
        labels = []
        for img, target in self.trainstream:
            kp = self.ORB.detect(img, None)
            _, des = self.ORB.compute(img, kp)
            descriptors.append(des)
            labels.append(labels)
        return descriptors, labels


    def standardize(self):
        scale = StandardScaler().fit(self.hist)
        self.hist = scale.transform(self.hist)

    def predict(self, img):
        # Get keypoints and descriptors
        kp = self.ORB.detect(img, None)
        _, des = self.ORB.compute(img, kp)

        # Compute the visual document, or which words are in the picture
        visual_doc = self.kMeans.predict(des)

        # Construct a histogram from the visual document
        hist = np.zeros([VOCAB_SIZE])
        for w in visual_doc:
            hist[w] += 1

        # Use the classifier to predict the class
        return self.svm.predict(hist)




if __name__ == '__main__':
    bov = BOV('./lists/splits/0_train.txt','./lists/splits/0_test.txt')
    bov.train()
    res = bov.test()

    print(accuracy(res))

