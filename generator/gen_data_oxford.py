
""" Offline data generation for the OXFORD dataset."""


import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import cv2
import os, glob
import argparse


SEQ_LENGTH = 3
WIDTH = 416
HEIGHT = 128
STEPSIZE = 1
CROP_AREA = [0, 360, 1280, 730]
INPUT_DIR = '/media/RAIDONE/radice'
OUTPUT_DIR = '/media/RAIDONE/radice/STRUCT2DEPTH'
OXFORD_CALIB = '/media/RAIDONE/radice/OXFORD/calib_cam_to_cam.txt'
SEG_DIR = '/media/RAIDONE/radice/STRUCT2DEPTH'


def parse_args():
    parser = argparse.ArgumentParser(description='Data generator for depth-and-motion-learning')
    parser.add_argument('--folder', type=str,
                        help='folder containing files',
                        required=True)
    return parser.parse_args()


def run_all(args):
    folder = args.folder
    dataset = 'OXFORD'
    ct = 0
    input_path = os.path.join(INPUT_DIR, dataset, folder)

    # rename input files with leading zeros
    left_folder = os.path.join(input_path, 'processed/stereo/left')
    print(left_folder)
    right_folder = os.path.join(input_path, 'processed/stereo/right')
    print(right_folder)

    for file in os.listdir(left_folder):
        num = file.split('.')[0]
        num = num.zfill(10)
        new_filename = num + '.jpg'
        os.rename(os.path.join(left_folder, file), os.path.join(left_folder, new_filename))

    for file in os.listdir(right_folder):
        num = file.split('.')[0]
        num = num.zfill(10)
        new_filename = num + '.jpg'
        os.rename(os.path.join(right_folder, file), os.path.join(right_folder, new_filename))

    # start processing
    print('-> Processing', input_path)
    output_path = os.path.join(OUTPUT_DIR, dataset, folder)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        left = os.path.join(output_path, 'left')
        os.mkdir(left)
        right = os.path.join(output_path, 'right')
        os.mkdir(right)

    # oxford calib matrix
    fx = 983.044006
    fy = 983.044006
    cx = 643.646973
    cy = 493.378998

    crop_height = CROP_AREA[3] - CROP_AREA[1]
    crop_ci = CROP_AREA[3] - (crop_height / 2)
    crop_cy  = cy + float(crop_height - 1) / 2 - crop_ci

    # calib_camera = np.array([[fx, 0.0, cx],
    #                           [0.0, fy, crop_cy],
    #                           [0.0, 0.0, 1.0]])

    for subfolder in ['processed/stereo/left', 'processed/stereo/right']:
        ct = 1
        # conversione in jpg
        files_path = os.path.join(input_path, subfolder)
        files = glob.glob(files_path + '/*.jpg')
        files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
        files = sorted(files)
        for i in range(SEQ_LENGTH, len(files)+1, STEPSIZE):
            imgnum = str(ct).zfill(10)

            # if os.path.exists(output_path + '/' + imgnum + '.png'):
            #     ct+=1
            #     continue
            big_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))
            big_seg_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))
            wct = 0

            for j in range(i-SEQ_LENGTH, i):  # Collect frames for this sample.
                img = cv2.imread(files[j])
                if subfolder == 'processed/stereo/left':
                    seg_path = os.path.join(SEG_DIR, dataset, folder, 'masks', 'left',
                                            os.path.basename(files[j]).replace('.jpg', '-fseg.png'))
                else:
                    seg_path = os.path.join(SEG_DIR, dataset, folder, 'masks', 'right',
                                            os.path.basename(files[j]).replace('.jpg', '-fseg.png'))

                segimg = cv2.imread(seg_path)

                ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape

                img = img[CROP_AREA[1]:CROP_AREA[3], :, :]

                zoom_x = WIDTH / ORIGINAL_WIDTH
                zoom_y = HEIGHT / ORIGINAL_HEIGHT

                # Adjust intrinsics.
                # calib_current = calib_camera.copy()
                # calib_current[0, 0] *= zoom_x
                # calib_current[0, 2] *= zoom_x
                # calib_current[1, 1] *= zoom_y
                # calib_current[1, 2] *= zoom_y

                current_fx = fx * zoom_x
                current_fy = fy * zoom_y
                current_cx = cx * zoom_x
                current_cy = crop_cy * zoom_y
                calib_current = np.array([[current_fx, 0.0, current_cx],
                                         [0.0, current_fy, current_cy],
                                         [0.0, 0.0, 1.0]])

                calib_representation = ','.join([str(c) for c in calib_current.flatten()])
                img = cv2.resize(img, (WIDTH, HEIGHT))
                segimg = cv2.resize(segimg, (WIDTH, HEIGHT))
                big_img[:,wct*WIDTH:(wct+1)*WIDTH] = img
                big_seg_img[:,wct*WIDTH:(wct+1)*WIDTH] = segimg
                wct+=1

            if subfolder == 'processed/stereo/left':
                big_img_path = os.path.join(output_path, 'left', '{}.{}'.format(imgnum, 'png'))
                txt_path = os.path.join(output_path, 'left', '{}{}.{}'.format(imgnum, '_cam', 'txt'))
                big_seg_img_path = os.path.join(output_path, 'left', '{}{}.{}'.format(imgnum, '-fseg', 'png'))
            else:
                big_img_path = os.path.join(output_path, 'right', '{}.{}'.format(imgnum, 'png'))
                txt_path = os.path.join(output_path, 'right', '{}{}.{}'.format(imgnum, '_cam', 'txt'))
                big_seg_img_path = os.path.join(output_path, 'right', '{}{}.{}'.format(imgnum, '-fseg', 'png'))

            cv2.imwrite(big_img_path, big_img)
            cv2.imwrite(big_seg_img_path, big_seg_img)
            f = open(txt_path, 'w')
            f.write(calib_representation)
            f.close()
            ct+=1

    print('-> DONE')


def main(args):
    run_all(args)


if __name__ == '__main__':
    args = parse_args()
    app.run(main(args))