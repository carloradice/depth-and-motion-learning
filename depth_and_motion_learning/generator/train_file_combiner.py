"""
Combina gli split di diverse run.
"""


import os
import random
import argparse

DIR = '/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/splits'

def parse_args():
    parser = argparse.ArgumentParser(description='Combines different routes')

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
        train.extend(lines[100:(len(lines) - 500)])
        file.close()
        test.extend(lines[(len(lines) - 500):])

        folders_names += folder + '_'


    # Shuffle e subsample se le liste sono troppo grandi
    random.shuffle(train)
    cut = 50000
    if len(train) > cut:
        train = random.sample(train, cut)

    random.shuffle(test)
    cut = 1000
    if len(test) > cut:
        test = random.sample(test, cut)


    splits_path = os.path.join(DIR, dataset, 'combined')
    splits = os.path.join(splits_path, folders_names + '{}_files.txt')

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