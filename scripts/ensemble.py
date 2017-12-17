import collections
import random

import cv2

from bf import BF
from preprocess import resize, grayscale, chop_lower
from util import data_loader, fscore2, class_report2

NUM_MATCHES = 30


# In the future, could add a params option for separate params for each
class Ensemble(object):
    def __init__(self, trainpath, testpath, detectors=('surf', 'sift', 'orb'), weights=None, params=None):
        if weights:
            assert len(detectors) == len(weights)
            self.weights = weights
        else:
            self.weights = [1] * len(detectors)

        self.trainpath = trainpath
        self.testpath = testpath

        self.detectors = detectors
        self.members = []
        for detector in detectors:
            self.members.append(BF(trainpath,
                                   testpath,
                                   n=1,
                                   detector=detector,
                                   use_n_matches=NUM_MATCHES))

        self.labels = [l[1] for l in data_loader(filepath=testpath)]

    def train(self):
        for member in self.members:
            member.train()

    def test(self):
        voices = []
        for member in self.members:
            preds, _ = member.test()
            voices.append(preds)
        harmony = self.cacophony(voices)
        return harmony, self.labels

    def cacophony(self, voices):
        harmony = []
        for notes in zip(voices):
            c = collections.Counter()
            for i, note in enumerate(notes):
                c[note] += self.weights[i]
            harmony.append(max(c.keys(), key=lambda x: c[x]))

        return harmony

    def limited_memory(self):
        voices = []
        print(self.members)

        del self.members

        for detector in self.detectors:
            member = BF(self.trainpath,
                        self.testpath,
                        n=1,
                        detector=detector,
                        use_n_matches=NUM_MATCHES)
            print(detector)
            member.train()
            preds, _ = member.test()
            voices.append(preds)
            print(preds)
            del member

        harmony = self.cacophony(voices)
        return harmony, self.labels


if __name__ == '__main__':
    split = random.randint(0, 9)

    ensemble = Ensemble('./lists/splits/{}_train.txt'.format(split),
                        './lists/splits/{}_test.txt'.format(split),
                        detectors=('sift', 'surf', 'orb'),
                        weights=None)

    preds, labels = ensemble.limited_memory()

    print(fscore2(labels, preds))
    print(class_report2(labels, preds))
