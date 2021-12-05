# Train file generator
# Each line is
# folder file_name

import argparse
import glob
import os
import random

INPUT_DIR = '/media/RAIDONE/radice/datasets'
OUTPUT_DIR = '/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/splits'


def parse_args():
    parser = argparse.ArgumentParser(description='Train file generator for depth-and-motion-learning')
    parser.add_argument('--folder', type=str,
                        help='folder containing train files',
                        required=True)
    parser.add_argument('--dataset', type=str,
                        help='dataset',
                        choices=['kitti', 'oxford'])
    return parser.parse_args()


def write_txt(file, path):
    """
    Write file to path
    """
    f = open(path, 'w')
    f.writelines(file)
    f.close()


def generator(args):
    folder = args.folder
    dataset = args.dataset
    if dataset == 'oxford':
        input_path = os.path.join(INPUT_DIR, dataset, folder, 'struct2depth')
        output_path = os.path.join(OUTPUT_DIR, dataset, folder)

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        # HARD CODED
        path = os.path.join(input_path, 'left')
        print('-> Examples path', path)
        files = glob.glob(path + '/*[0-9].png')
        files = sorted(files)
        print('-> Number of examples before cut', len(files))
        files = files[100:(len(files)-100)]
        print('-> Number of examples after cut', len(files))

        pairs = []
        for index, f in enumerate(files):
            if index == len(files)-1:
                pairs.append(path + ' ' + os.path.basename(f).split('.')[0])
            else:
                pairs.append(path + ' ' + os.path.basename(f).split('.')[0] + '\n')

        # divisione train/test
        cut = int(len(pairs) * 0.9) + 1
        train = pairs[:cut]
        test = pairs[cut:]

        splits = os.path.join(output_path, '{}.txt')

        # lista dei files in ordine
        write_txt(pairs, splits.format('examples'))

        # train
        print('-> Number of examples for training', len(train))
        # Shuffle
        random.shuffle(train)
        write_txt(train, splits.format('train'))

        # test
        print('-> Number of examples for testing', len(test))
        if len(test) > 100:
            print('-> Subsample examples for testing')
            N = 2
            subsample = test[::N]
            while len(subsample) > 100:
                subsample = test[::N]
                N += 2
            test = subsample
        print('-> Number of examples for testing', len(test))
        write_txt(test, splits.format('test'))


if __name__ == '__main__':
    args = parse_args()
    generator(args)
