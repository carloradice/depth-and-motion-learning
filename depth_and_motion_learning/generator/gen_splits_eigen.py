import os
import random

DEFAULT_PATH = '/media/RAIDONE/radice/datasets/kitti/struct2depth'
INPUT_DIR = '/media/RAIDONE/radice/datasets/kitti/'
OUTPUT_DIR = '/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/splits/kitti/'


def fill_data(lines):
    data = []
    for line in lines:
        path = line.rstrip().split()[0]
        path = path.split('/')

        date = path[0]
        sequence = path[1]
        # folder = 'image_02' or 'image_s03'
        basename = path[4].split('.')[0]

        l = os.path.join(DEFAULT_PATH, date, sequence, 'image_02') + ' ' + '{}{}'.format(basename, '\n')
        r = os.path.join(DEFAULT_PATH, date, sequence, 'image_03') + ' ' + '{}{}'.format(basename, '\n')

        data.append(l)
        data.append(r)

    return data


def gen_splits():
    # Train
    file = open(INPUT_DIR + 'eigen_train_files.txt', 'r')
    lines = file.readlines()
    train = fill_data(lines=lines)
    random.shuffle(train)
    file = open(OUTPUT_DIR + 'eigen_train_files.txt', 'w')
    file.writelines(train)

    # Test
    file = open(INPUT_DIR + 'eigen_test_files.txt', 'r')
    lines = file.readlines()
    test = fill_data(lines=lines)
    random.shuffle(test)
    file = open(OUTPUT_DIR + 'eigen_test_files.txt', 'w')
    file.writelines(test)


if __name__ == '__main__':
    gen_splits()