#! /usr/bin/python3
import os
import random
random.seed()

def generate_splits(n=10,split=.8):
    data = [ line.strip('\n').split(',') for line in open('./lists/skyline_data.txt')]
    split_index = int(len(data) * split)
    for i in range(n):
        random.shuffle(data)
        train = data[:split_index]
        test = data[split_index:]
        with open('./lists/splits/{}_train.txt'.format(i),'w+') as f:
            for path, label in train:
                f.write('{},{}\n'.format(path,label))

        with open('./lists/splits/{}_test.txt'.format(i),'w+') as f:
            for path, label in test:
                f.write('{},{}\n'.format(path,label))

def generate_vgg_splits(n=10,split=.8, folder=None, listpath=None):
    try:
        os.mkdir(folder + '/splits')
    except OSError:
        pass
    data = [ line.strip('\n').split(',') for line in open(listpath)]
    split_index = int(len(data) * split)
    for i in range(n):
        random.shuffle(data)
        train = data[:split_index]
        test = data[split_index:]
        with open(folder + '/splits/{}_train.txt'.format(i),'w+') as f:
            for path, label in train:
                f.write('{} {}\n'.format(path,label))

        with open(folder + '/splits/{}_test.txt'.format(i),'w+') as f:
            for path, label in test:
                f.write('{} {}\n'.format(path,label))


def generate_full_list():
    items = ['./lists/chicago_out.txt','./lists/london_out.txt','./lists/nyc_out.txt','./lists/shanghai_out.txt']
    out = []
    for i in items:
        out += [line for line in open(i).readlines()]

    with open('skyline_data.txt','w+') as f:
        for o in out:
            f.write(o)


if __name__ == '__main__':
    import sys

    try:
        # generate_splits(sys.argv[0],sys.argv[1])
        generate_vgg_splits(sys.argv[0],sys.argv[1])
    except IndexError:
        print('No arguments, using defaults (generate_splits n_splits ratio)')
        generate_vgg_splits()
        # generate_splits()
