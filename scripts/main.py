#! /usr/bin/env/python

import scripts.data_loader as loader
import scripts.preprocess as preprocess
import scripts.util


def train(images):
    pass

def test(model, images):
    pass

if __name__ == '__main__':

    for i in range(0,10):
        images = scripts.util.data_loader()
        model = train(images)
        result = test(model, images)