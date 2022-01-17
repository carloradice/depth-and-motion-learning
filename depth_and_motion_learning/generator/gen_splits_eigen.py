import os
import random
import argparse

INPUT_DIR = '/media/RAIDONE/radice/datasets/kitti/'
OUTPUT_DIR = '/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/splits/kitti/'


def parse_args():
    parser = argparse.ArgumentParser(description='Splits generator for kitti')
    parser.add_argument('--folder', type=str,
                        help='folder containing files',
                        required=True)
    return parser.parse_args()


def fill_data(lines, type, folder=None):
    data = []
    for line in lines:
        path = line.rstrip().split()[0]
        path = path.split('/')

        date = path[0]
        sequence = path[1]
        # folder = 'image_02' or 'image_03'
        basename = path[4].split('.')[0]

        # Se esiste il file, aggiungerlo allo split
        # Ci sono meno di 22600 * 2 files poichè se il file è il primo o l'ultimo di una sequenza non viene
        # considerato in quanto la tripletta non avrebbe una delle 3 immagini
        if type == 'train':
            f = os.path.join(INPUT_DIR, folder, date, sequence, 'image_02', '{}.png'.format(basename))
            if os.path.isfile(f):
                l = os.path.join(INPUT_DIR, folder, date, sequence, 'image_02') + ' ' + '{}{}'.format(basename, '\n')
                r = os.path.join(INPUT_DIR, folder, date, sequence, 'image_03') + ' ' + '{}{}'.format(basename, '\n')
                data.append(l)
                data.append(r)

        if type == 'test':
            l = os.path.join(INPUT_DIR, 'data', date, sequence, 'image_02', 'data') + ' ' + '{}{}'.format(basename, '\n')
            r = os.path.join(INPUT_DIR, 'data', date, sequence, 'image_03', 'data') + ' ' + '{}{}'.format(basename, '\n')
            data.append(l)
            data.append(r)


    return data


def gen_splits():

    args = parse_args()
    folder = args.folder
    sub = folder.split('-', 1)[1:][0]

    # Train
    file = open(INPUT_DIR + 'eigen_train_files.txt', 'r')
    lines = file.readlines()
    print(len(lines))
    train = fill_data(lines=lines, type='train', folder=folder)
    print(len(train))
    random.shuffle(train)
    file = open(OUTPUT_DIR + 'eigen_train_files_' + sub + '.txt', 'w')
    file.writelines(train)

    # Test
    file = open(INPUT_DIR + 'eigen_test_files.txt', 'r')
    lines = file.readlines()
    test = fill_data(lines=lines, type='test')
    random.shuffle(test)
    file = open(OUTPUT_DIR + 'eigen_test_files.txt', 'w')
    file.writelines(test)


if __name__ == '__main__':
    gen_splits()