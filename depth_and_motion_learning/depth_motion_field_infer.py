# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""A binary for training depth and egomotion."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import flags
from absl import app

from depth_and_motion_learning import depth_motion_field_model
from depth_and_motion_learning import training_utils
from depth_and_motion_learning import parameter_container

import cv2
import numpy as np
import os

FLAGS = flags.FLAGS
WIDTH = 416
HEIGHT = 128
CROP_AREA = [0, 360, 1280, 730]
OUTPUT_DIR = '/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/inference'
DATASET = 'kitti'


def save_depth(file, save_path):
    example_save_path = os.path.join(save_path, '{}.png')

    for line in file:

        line = line.rstrip().split(' ')
        example = os.path.join(line[0], '{}.{}'.format(line[1], 'png'))

        image = cv2.imread(example).astype(np.float32)
        image = image[CROP_AREA[1]:CROP_AREA[3], :, :]

        # save cropped image per confronto
        cv2.imwrite(example_save_path.format(line[1]), image)

        # image = cv2.resize(image, (416, 128))
        # input_batch = np.reshape(image, (1, 128, 416, 3))

        image = cv2.resize(image, (WIDTH, HEIGHT))
        input_batch = np.reshape(image, (1, HEIGHT, WIDTH, 3))

        depth = training_utils.infer(depth_motion_field_model.input_fn_infer(input_image=input_batch),
                                     depth_motion_field_model.loss_fn,
                                     depth_motion_field_model.get_vars_to_restore_fn)

        depth.figure.savefig(example_save_path.format(line[1] + '_depth'))


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    model_dir = FLAGS.model_dir

    # Dynamic examples
    params = parameter_container.ParameterContainer({
      'model': {
          'batch_size': 16,
          'input': {}
      },
    }, {'trainer': {
      'master': FLAGS.master,
      'model_dir': model_dir
    }})

    model = os.path.basename(model_dir)

    params.override(FLAGS.param_overrides)
    input = params.model.get('input')
    test_file_path = input.get('data_path')
    dataset = input.get('dataset')

    test_file = open(test_file_path, 'r')
    lines = test_file.readlines()
    test_file.close()

    if DATASET == 'oxford':

        folder = (lines[0].split(' '))[0].split('/')[6]

        save_path = os.path.join(OUTPUT_DIR, 'oxford', model, folder)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        examples_save_path = os.path.join(save_path, 'dynamic_examples')
        if not os.path.isdir(examples_save_path):
            os.makedirs(examples_save_path)

        save_depth(file=lines, save_path=examples_save_path)


    if DATASET == 'kitti':

        model = model.split('-', 1)[1:][0]

        for line in lines:

            line = line.split(' ')

            date = line[0].split('/')[7]
            seqname = line[0].split('/')[8]
            subfolder = line[0].split('/')[9]

            line[1] = line[1].rstrip()
            example = os.path.join(line[0], '{}.png'.format(line[1]))

            image = cv2.imread(example).astype(np.float32)

            examples_save_path = os.path.join(OUTPUT_DIR, 'kitti', model, date, seqname, subfolder)
            test_images_save_path = os.path.join(OUTPUT_DIR, 'kitti', 'test-images', date, seqname, subfolder)
            if not os.path.isdir(examples_save_path):
                os.makedirs(examples_save_path)
            if not os.path.isdir(test_images_save_path):
                os.makedirs(test_images_save_path)
            # save image per confronto
            if not os.path.isfile(os.path.join(test_images_save_path, '{}.png'.format(line[1]))):
                cv2.imwrite(os.path.join(test_images_save_path, '{}.png'.format(line[1])), image)

            image = cv2.resize(image, (WIDTH, HEIGHT))
            input_batch = np.reshape(image, (1, HEIGHT, WIDTH, 3))

            depth_ax, depth = training_utils.infer(depth_motion_field_model.input_fn_infer(input_image=input_batch),
                                         depth_motion_field_model.loss_fn,
                                         depth_motion_field_model.get_vars_to_restore_fn)

            # save depth .png
            depth_ax.figure.savefig(os.path.join(examples_save_path, '{}_depth.png'.format(line[1])))
            # save depth .npy
            np.save(os.path.join(examples_save_path, '{}_depth.npy'.format(line[1])), depth)


if __name__ == '__main__':
  app.run(main)
