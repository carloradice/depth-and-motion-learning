# Train file generator
# Each line is
# folder file_name

import argparse
import glob
import os

INPUT_DIR = '/media/RAIDONE/radice/STRUCT2DEPTH'
OUTPUT_DIR = '/home/radice/neuralNetworks/depth_and_motion_learning/splits'


def parse_args():
    parser = argparse.ArgumentParser(description='Train file generator for depth-and-motion-learning')
    parser.add_argument('--folder', type=str,
                        help='folder containing train files',
                        required=True)
    parser.add_argument('--dataset', type=str,
                        help='dataset',
                        choices=['KITTI', 'OXFORD'])
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
    if dataset == 'OXFORD':
        input_path = os.path.join(INPUT_DIR, dataset, folder)
        output_path = os.path.join(OUTPUT_DIR, dataset, folder)

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        # HARD CODED
        path = os.path.join(input_path, 'left')
        files = glob.glob(path + '/*[0-9].png')
        print(files)
        files = sorted(files)

        names = []
        for index, f in enumerate(files):
            if index == len(files)-1:
                names.append(path + ' ' + os.path.basename(f).split('.')[0])
            else:
                names.append(path + ' ' + os.path.basename(f).split('.')[0] + '\n')

        train_path = os.path.join(output_path, '{}.{}'.format('train', 'txt'))
        write_txt(names, train_path)


if __name__ == '__main__':
    args = parse_args()
    generator(args)

