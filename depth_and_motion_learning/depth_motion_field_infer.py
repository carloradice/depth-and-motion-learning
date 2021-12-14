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
CROP_AREA = [0, 360, 1280, 730]
OUTPUT_DIR = '/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/inference/oxford'


def save_depth(file, save_path, flag=None):
    example_save_path = os.path.join(save_path, '{}.png')
    if not flag == None:

        for line in file:

            line = line.rstrip().split(' ')
            example = os.path.join(line[0], '{}.{}'.format(line[1], 'png'))

            image = cv2.imread(example).astype(np.float32)
            image = image[CROP_AREA[1]:CROP_AREA[3], :, :]

            # save cropped image per confronto
            cv2.imwrite(example_save_path.format(line[1]), image)

            # image = cv2.resize(image, (416, 128))
            # input_batch = np.reshape(image, (1, 128, 416, 3))

            image = cv2.resize(image, (640, 192))
            input_batch = np.reshape(image, (1, 192, 640, 3))

            depth = training_utils.infer(depth_motion_field_model.input_fn_infer(input_image=input_batch),
                                         depth_motion_field_model.loss_fn,
                                         depth_motion_field_model.get_vars_to_restore_fn)

            depth.figure.savefig(example_save_path.format(line[1] + '_depth'))

    else:

        for line in file:
            image = cv2.imread(line).astype(np.float32)
            image = image[CROP_AREA[1]:CROP_AREA[3], :, :]

            # save cropped image per confronto
            name = os.path.basename(line).split('.')[0]
            cv2.imwrite(example_save_path.format(name), image)

            # image = cv2.resize(image, (416, 128))
            # input_batch = np.reshape(image, (1, 128, 416, 3))

            image = cv2.resize(image, (640, 192))
            input_batch = np.reshape(image, (1, 192, 640, 3))

            depth = training_utils.infer(depth_motion_field_model.input_fn_infer(input_image=input_batch),
                                         depth_motion_field_model.loss_fn,
                                         depth_motion_field_model.get_vars_to_restore_fn)

            depth.figure.savefig(example_save_path.format(name + '_depth'))


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
    dynamic_test_file_path = input.get('data_path')

    dynamic_test_file = open(dynamic_test_file_path, 'r')
    lines = dynamic_test_file.readlines()
    dynamic_test_file.close()

    folder = (lines[0].split(' '))[0].split('/')[6]

    save_path = os.path.join(OUTPUT_DIR, model, folder)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    dynamic_examples_save_path = os.path.join(save_path, 'dynamic_examples')
    if not os.path.isdir(dynamic_examples_save_path):
        os.makedirs(dynamic_examples_save_path)

    save_depth(file=lines, save_path=dynamic_examples_save_path, flag=1)

    # Default examples
    # default_test_files = ['/media/RAIDONE/radice/2014-06-26-09-31-18/processed/stereo/left/633.jpg']
    #
    # default_examples_save_path = os.path.join(save_path, 'default_examples')
    # if not os.path.isdir(default_examples_save_path):
    #     os.makedirs(default_examples_save_path)
    #
    # save_depth(file=default_test_files, save_path=default_examples_save_path)


if __name__ == '__main__':
  app.run(main)
