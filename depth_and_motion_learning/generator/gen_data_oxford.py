
""" Offline data generation for the OXFORD dataset."""

from absl import app
import numpy as np
import cv2
import os, glob
import argparse
# run time
import timeit
# time format
import time
from datetime import datetime

SEQ_LENGTH = 3

# WIDTH = 640
# HEIGHT = 192

# WIDTH = 416
# HEIGHT = 128

# cerco di mantenere le proporzioni simili alla dimensione di crop
WIDTH = 416
HEIGHT = 198
STEPSIZE = 1
# CROP_AREA = [0, 360, 1280, 730]
CROP_AREA = [0, 200, 1280, 810]
# mask-rcnn score limit
LIMIT = 80
DIR = '/media/RAIDONE/radice/datasets/oxford'
STRUCT2DEPTH_FOLDER = 'struct2depth-80-416x198'

def parse_args():
    parser = argparse.ArgumentParser(description='Data generator for depth-and-motion-learning')
    parser.add_argument('--folder', type=str,
                        help='folder containing files',
                        required=True)
    return parser.parse_args()

def get_camera_intrinsics_matrix():
    # original dimensions
    original_width = 1280
    original_height = 960
    # original oxford intrinsic matrix parameters
    fx = 983.044006
    fy = 983.044006
    cx = 643.646973
    cy = 493.378998

    # new cy after crop
    # crop_height = CROP_AREA[3] - CROP_AREA[1]
    # crop_ci = CROP_AREA[3] - (crop_height / 2)
    # crop_cy = cy + float(crop_height - 1) / 2 - crop_ci
    crop_cy = cy - CROP_AREA[1]

    # scales
    scale_x = WIDTH / original_width
    scale_y = HEIGHT / original_height

    # new parameters after resize
    current_fx = fx * scale_x
    current_fy = fy * scale_y
    current_cx = cx * scale_x
    current_cy = crop_cy * scale_y

    # intrinsics_matrix = np.array([[fx, 0.0, cx],
    #                          [0.0, fy, cy],
    #                          [0.0, 0.0, 1.0]])

    intrinsics_matrix = np.array([[current_fx, 0.0, current_cx],
                                  [0.0, current_fy, current_cy],
                                  [0.0, 0.0, 1.0]])

    return intrinsics_matrix


def run_all(args):
    folder = args.folder
    path = os.path.join(DIR, folder)

    print('-> Parameters:\n WIDTH={},\n HEIGTH={},\n CROP_AREA={},\n LIMIT={}'.format(WIDTH, HEIGHT, CROP_AREA, LIMIT))

    # start processing
    print('-> Processing sequence', folder)

    struct2depth_path = os.path.join(DIR, folder, STRUCT2DEPTH_FOLDER)

    if not os.path.exists(struct2depth_path):
        os.makedirs(struct2depth_path)

    if not os.path.exists(os.path.join(struct2depth_path, 'left')):
        os.mkdir(os.path.join(struct2depth_path, 'left'))
    if not os.path.exists(os.path.join(struct2depth_path, 'right')):
        os.mkdir(os.path.join(struct2depth_path, 'right'))

    intrinsics_matrix = get_camera_intrinsics_matrix()
    calib_representation = ','.join([str(c) for c in intrinsics_matrix.flatten()])

    for subfolder in ['stereo/left', 'stereo/right']:

        start_partial = timeit.default_timer()
        current_seg = start_partial

        ct = 0
        files_path = os.path.join(path, subfolder)
        files = glob.glob(files_path + '/*.png')
        files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
        files = sorted(files)

        for i in range(SEQ_LENGTH, len(files)+1, STEPSIZE):
            imgnum = str(ct).zfill(10)

            big_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))
            big_seg_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))
            wct = 0

            for j in range(i-SEQ_LENGTH, i):  # Collect frames for this sample.
                img = cv2.imread(files[j])
                img = img[CROP_AREA[1]:CROP_AREA[3], :, :]

                if subfolder == 'stereo/left':
                    seg_path = os.path.join(path, 'rcnn-masks-classes', 'left',
                                            os.path.basename(files[j]).replace('.png', '.npz'))
                else:
                    seg_path = os.path.join(path, 'rcnn-masks-classes', 'right',
                                            os.path.basename(files[j]).replace('.png', '.npz'))

                l = np.load(seg_path, allow_pickle=True)
                segdict = l['arr_0'].item()
                segimg = np.zeros([segdict['score_mask'].shape[0], segdict['score_mask'].shape[1], 3])
                # class_names = ['BG'=0, 'person'=1, 'bicycle'=2, 'car'=3, 'motorcycle'=4, 'bus'=6 'truck'=8]
                index_array = np.where((segdict['score_mask'] >= LIMIT) &
                                       ((segdict['class_ids'] == 1) | (segdict['class_ids'] == 2) |
                                        (segdict['class_ids'] == 3) | (segdict['class_ids'] == 4) |
                                        (segdict['class_ids'] == 6) | (segdict['class_ids'] == 8)))
                for i in range(len(index_array[0])):
                    segimg[index_array[0][i], index_array[1][i], :] = 255

                img = cv2.resize(img, (WIDTH, HEIGHT))
                segimg = cv2.resize(segimg, (WIDTH, HEIGHT))
                big_img[:,wct*WIDTH:(wct+1)*WIDTH] = img
                big_seg_img[:,wct*WIDTH:(wct+1)*WIDTH] = segimg
                wct+=1

            if subfolder == 'stereo/left':
                big_img_path = os.path.join(struct2depth_path, 'left', '{}.{}'.format(imgnum, 'png'))
                txt_path = os.path.join(struct2depth_path, 'left', '{}{}.{}'.format(imgnum, '_cam', 'txt'))
                big_seg_img_path = os.path.join(struct2depth_path, 'left', '{}{}.{}'.format(imgnum, '-fseg', 'png'))
            else:
                big_img_path = os.path.join(struct2depth_path, 'right', '{}.{}'.format(imgnum, 'png'))
                txt_path = os.path.join(struct2depth_path, 'right', '{}{}.{}'.format(imgnum, '_cam', 'txt'))
                big_seg_img_path = os.path.join(struct2depth_path, 'right', '{}{}.{}'.format(imgnum, '-fseg', 'png'))

            cv2.imwrite(big_img_path, big_img)
            cv2.imwrite(big_seg_img_path, big_seg_img)
            f = open(txt_path, 'w')
            f.write(calib_representation)
            f.close()

            if ct%1000==0 and ct!=0:
                print('->', ct, 'Done')
                stop_seg = timeit.default_timer()
                seg_run_time = int(stop_seg - current_seg)
                print('-> Segment run time:', time.strftime('%H:%M:%S', time.gmtime(seg_run_time)))
                current_seg += seg_run_time

            ct+=1

        stop_partial = timeit.default_timer()
        partial_run_time = int(stop_partial - start_partial)
        print('-> Partial run time:', time.strftime('%H:%M:%S', time.gmtime(partial_run_time)))

    print('-> DONE')


def main(args):
    run_all(args)


if __name__ == '__main__':
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("-> Start:", dt_string)

    # start timer
    start = timeit.default_timer()

    args = parse_args()
    app.run(main(args))

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("-> End:", dt_string)

    # stop timer
    stop = timeit.default_timer()

    # total run time
    total_run_time = int(stop - start)
    print('-> Total run time:', time.strftime('%H:%M:%S', time.gmtime(total_run_time)))

