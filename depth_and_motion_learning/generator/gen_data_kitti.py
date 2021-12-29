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

from absl import app
import numpy as np
import cv2
import os, glob
import timeit, time

SEQ_LENGTH = 3
WIDTH = 640
HEIGHT = 192
STEPSIZE = 1
# mask-rcnn score limit
LIMIT = 90
INPUT_DIR = '/media/RAIDONE/radice/datasets/kitti'
OUTPUT_DIR = '/media/RAIDONE/radice/datasets/kitti/struct2depth'

if not OUTPUT_DIR.endswith('/'):
    OUTPUT_DIR = OUTPUT_DIR + '/'


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


def run_all():
    partial_run_time = 0

    for d in glob.glob(INPUT_DIR + '/data' + '/*/'):

        date = d.split('/')[-2]
        file_calibration = d + 'calib_cam_to_cam.txt'
        calib_raw = [get_line(file_calibration, 'P_rect_02'), get_line(file_calibration, 'P_rect_03')]

        if not os.path.exists(os.path.join(OUTPUT_DIR, date)):
            os.makedirs(os.path.join(OUTPUT_DIR, date))

        for d2 in glob.glob(d + '*/'):
            seqname = d2.split('/')[-2]
            print('Processing sequence', seqname)

            half_path = os.path.join(OUTPUT_DIR, date, seqname)
            if not os.path.exists(half_path):
                os.mkdir(half_path)

            start_seg = timeit.default_timer()

            for subfolder in ['image_02/data', 'image_03/data']:
                ct = 1

                full_path = os.path.join(half_path, subfolder.replace('/data', ''))
                if not os.path.exists(full_path):
                    os.mkdir(full_path)

                calib_camera = calib_raw[0] if subfolder=='image_02/data' else calib_raw[1]
                folder = d2 + subfolder
                files = glob.glob(folder + '/*.png')
                files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
                files = sorted(files)
                for i in range(SEQ_LENGTH, len(files)+1, STEPSIZE):
                    imgnum = str(ct).zfill(10)
                    # if (os.path.isfile(os.path.join(full_path, '{}.png'.format(imgnum)))) and \
                    #         (os.path.isfile(os.path.join(full_path, '{}-fseg.png'.format(imgnum)))) and \
                    #         (os.path.isfile(os.path.join(full_path, '{}_cam.txt'.format(imgnum)))):
                    #     ct+=1
                    #     continue

                    big_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))
                    big_seg_img = np.zeros(shape=(HEIGHT, WIDTH * SEQ_LENGTH, 3))
                    wct = 0

                    for j in range(i-SEQ_LENGTH, i):  # Collect frames for this sample.
                        img = cv2.imread(files[j])
                        ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape

                        zoom_x = WIDTH/ORIGINAL_WIDTH
                        zoom_y = HEIGHT/ORIGINAL_HEIGHT

                        # Adjust intrinsics.
                        calib_current = calib_camera.copy()
                        calib_current[0, 0] *= zoom_x
                        calib_current[0, 2] *= zoom_x
                        calib_current[1, 1] *= zoom_y
                        calib_current[1, 2] *= zoom_y

                        calib_representation = ','.join([str(c) for c in calib_current.flatten()])

                        # Load mask for current file
                        seg_path = os.path.join(INPUT_DIR, 'mask-rcnn', date, seqname, subfolder.replace('/data', ''),
                                                os.path.basename(files[j]).replace('.png', '.npz'))
                        l = np.load(seg_path, allow_pickle=True)
                        segdict = l['arr_0'].item()
                        segimg = np.zeros([segdict['score_mask'].shape[0], segdict['score_mask'].shape[1], 3])
                        index_array = np.where(segdict['score_mask']>=LIMIT)
                        for i in range(len(index_array[0])):
                            segimg[index_array[0][i], index_array[1][i], :] = 255

                        img = cv2.resize(img, (WIDTH, HEIGHT))
                        big_img[:,wct*WIDTH:(wct+1)*WIDTH] = img

                        segimg = cv2.resize(segimg, (WIDTH, HEIGHT))
                        big_seg_img[:, wct * WIDTH:(wct + 1) * WIDTH] = segimg

                        wct+=1

                    cv2.imwrite(os.path.join(full_path, '{}.png'.format(imgnum)), big_img)
                    cv2.imwrite(os.path.join(full_path, '{}-fseg.png'.format(imgnum)), big_seg_img)
                    f = open(os.path.join(full_path, '{}_cam.txt'.format(imgnum)), 'w')
                    f.write(calib_representation)
                    f.close()

                    ct += 1

            stop_seg = timeit.default_timer()
            seg_run_time = int(stop_seg - start_seg)
            partial_run_time += seg_run_time
            print('-> Segment run time:', time.strftime('%H:%M:%S', time.gmtime(seg_run_time)))
            print('-> Partial run time:', time.strftime('%H:%M:%S', time.gmtime(partial_run_time)))


def main(_):
    #run_all()
    run_all()


if __name__ == '__main__':
    # start timer
    start = timeit.default_timer()

    app.run(main)

    # stop timer
    stop = timeit.default_timer()

    # total run time
    total_run_time = int(stop - start)
    print('-> Total run time:', time.strftime('%H:%M:%S', time.gmtime(total_run_time)))
