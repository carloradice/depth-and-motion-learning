"""
Unisce gli splits di diverse run.
"""

import os
import random
import argparse

DIR = '/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/splits'

def parse_args():
    parser = argparse.ArgumentParser(description='Mixes different routes')

    parser.add_argument('--list',
                        help='routes to combine',
                        nargs='+',
                        required=True)

    parser.add_argument('--dataset', type=str,
                        help='dataset',
                        choices=['oxford'])

    return parser.parse_args()


def combine(folders, dataset):
    """
    Combine multiple splits.
    """
    train = []
    test = []

    folders_names = ''

    for folder in folders:

        path = os.path.join(DIR, dataset, folder, 'train.txt')
        file = open(path, 'r')
        lines = file.readlines()
        file.close()
        train.extend(lines)

        path = os.path.join(DIR, dataset, folder, 'test.txt')
        file = open(path, 'r')
        lines = file.readlines()
        file.close()
        test.extend(lines)

        folders_names += folder + '_'

    # Shuffle e subsample se le liste sono troppo grandi
    random.shuffle(train)
    cut = 50000
    if len(train) > cut:
        train = random.sample(train, cut)

    random.shuffle(test)
    cut = 100
    if len(test) > cut:
        test = random.sample(test, cut)

    splits_path = os.path.join(DIR, dataset, 'mixed', folders_names)
    splits = os.path.join(splits_path, '{}.txt')

    print('-> Save splits in', splits_path)

    f = open(splits.format('train'), 'w')
    f.writelines(train)
    f.close()

    f = open(splits.format('test'), 'w')
    f.writelines(test)
    f.close()


def main(args):
    """
    Main.
    """
    folders = args.list
    dataset = args.dataset

    combine(folders=folders, dataset=dataset)

if __name__ == '__main__':
    args = parse_args()
    main(args)