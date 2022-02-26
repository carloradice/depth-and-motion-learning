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
OUTPUT_DIR = '/media/RAIDONE/radice/neural-networks-data/predictions'


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

    test_file = open(test_file_path, 'r')
    lines = test_file.readlines()
    test_file.close()

    for line in lines:
        ########################################### KITTI ##############################################################
        # line = line.split()[0]
        #
        # line = line.split('/')
        #
        # date = line[0]
        # seqname = line[1]
        # subfolder = line[2]
        # basename = line[4].split('.')[0]
        ################################################################################################################
        ########################################### KITTI360 ###########################################################
        line = line.rstrip()
        basename = os.path.basename(line).split('.')[0]
        ################################################################################################################

        example = os.path.join(line)
        if not os.path.isfile(example):
            raise Exception ('{} is not a file'.format(example))
        image = cv2.imread(example).astype(np.float32)

        examples_save_path = os.path.join(OUTPUT_DIR, 'dml-360{}'.format(model))

        if not os.path.isdir(examples_save_path):
            os.makedirs(examples_save_path)

        image = cv2.resize(image, (WIDTH, HEIGHT))
        input_batch = np.reshape(image, (1, HEIGHT, WIDTH, 3))

        depth_ax, depth = training_utils.infer(depth_motion_field_model.input_fn_infer(input_image=input_batch),
                                     depth_motion_field_model.loss_fn,
                                     depth_motion_field_model.get_vars_to_restore_fn)
        # save depth .png
        depth_ax.figure.savefig(os.path.join(examples_save_path, '{}_depth.png'.format(basename)))
        # save depth .npy
        np.save(os.path.join(examples_save_path, '{}_depth.npy'.format(basename)), depth)


if __name__ == '__main__':
  app.run(main)
