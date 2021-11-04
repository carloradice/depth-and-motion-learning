# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Offline data generation for the KITTI dataset."""

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
# INPUT_DIR = '/usr/local/google/home/anelia/struct2depth/KITTI_FULL/kitti-raw-uncompressed'
# OUTPUT_DIR = '/usr/local/google/home/anelia/struct2depth/KITTI_procesed/'
INPUT_DIR = '/media/RAIDONE/radice'
OUTPUT_DIR = '/media/RAIDONE/radice/STRUCT2DEPTH'
OXFORD_CALIB = '/media/RAIDONE/radice/OXFORD/calib_cam_to_cam.txt'


def parse_args():
    parser = argparse.ArgumentParser(description='Data generator for depth-and-motion-learning')
    parser.add_argument('--folder', type=str,
                        help='folder containing files',
                        required=True)
    parser.add_argument('--dataset', type=str,
                        help='dataset',
                        choices=['KITTI', 'OXFORD'])
    return parser.parse_args()


def get_line(file, start):
    file = open(file, 'r')
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    ret = None
    for line in lines:
        nline = line.split(': ')
        if nline[0]==start:
            ret = nline[1].split(' ')
            ret = np.array([float(r) for r in ret], dtype=float)
            ret = ret.reshape((3,4))[0:3, 0:3]
            break
    file.close()
    return ret


def crop(img, segimg, fx, fy, cx, cy):
    # Perform center cropping, preserving 50% vertically.
    middle_perc = 0.50
    left = 1-middle_perc
    half = left/2
    a = img[int(img.shape[0]*(half)):int(img.shape[0]*(1-half)), :]
    aseg = segimg[int(segimg.shape[0]*(half)):int(segimg.shape[0]*(1-half)), :]
    cy /= (1/middle_perc)

    # Resize to match target height while preserving aspect ratio.
    wdt = int((128*a.shape[1]/a.shape[0]))
    x_scaling = float(wdt)/a.shape[1]
    y_scaling = 128.0/a.shape[0]
    b = cv2.resize(a, (wdt, 128))
    bseg = cv2.resize(aseg, (wdt, 128))

    # Adjust intrinsics.
    fx*=x_scaling
    fy*=y_scaling
    cx*=x_scaling
    cy*=y_scaling

    # Perform center cropping horizontally.
    remain = b.shape[1] - 416
    cx /= (b.shape[1]/416)
    c = b[:, int(remain/2):b.shape[1]-int(remain/2)]
    cseg = bseg[:, int(remain/2):b.shape[1]-int(remain/2)]

    return c, cseg, fx, fy, cx, cy


def run_all(args):
    folder = args.folder
    dataset = args.dataset
    ct = 0
    if dataset == 'OXFORD':
        input_path = os.path.join(INPUT_DIR, dataset, folder)
        print('-> Processing', input_path)
        output_path = os.path.join(OUTPUT_DIR, dataset, folder)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # oxford calib matrix
        calib_camera = np.array([[983.044006, 0.0, 643.646973],
                             [0.0, 983.044006, 493.378998],
                             [0.0, 0.0, 1.0]])

        for subfolder in ['processed/stereo/left', 'processed/stereo/right']:
            ct = 1
            # conversione in jpg
            files_path = os.path.join(input_path, subfolder)
            files = glob.glob(files_path + '/*.jpg')
            files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
            files = sorted(files)
            for i in range(SEQ_LENGTH, len(files)+1, STEPSIZE):
                imgnum = str(ct).zfill(10)
                if os.path.exists(output_path + '/' + imgnum + '.png'):
                    ct+=1
                    continue
                big_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))
                wct = 0

                for j in range(i-SEQ_LENGTH, i):  # Collect frames for this sample.
                    img = cv2.imread(files[j])
                    ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape

                    zoom_x = WIDTH / ORIGINAL_WIDTH
                    zoom_y = HEIGHT / ORIGINAL_HEIGHT

                    # Adjust intrinsics.
                    calib_current = calib_camera.copy()
                    calib_current[0, 0] *= zoom_x
                    calib_current[0, 2] *= zoom_x
                    calib_current[1, 1] *= zoom_y
                    calib_current[1, 2] *= zoom_y

                    calib_representation = ','.join([str(c) for c in calib_current.flatten()])
                    img = cv2.resize(img, (WIDTH, HEIGHT))
                    big_img[:,wct*WIDTH:(wct+1)*WIDTH] = img
                    wct+=1
                cv2.imwrite(output_path + '/' + imgnum + '.png', big_img)
                f = open(output_path + '/' + imgnum + '_cam.txt', 'w')
                f.write(calib_representation)
                f.close()
                ct+=1

    print('-> DONE')


def main(args):
    run_all(args)


if __name__ == '__main__':
    args = parse_args()
    app.run(main(args))