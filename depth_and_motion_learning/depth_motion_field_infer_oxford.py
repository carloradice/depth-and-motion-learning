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
#CROP_AREA = [0, 360, 1280, 730]
OUTPUT_DIR = '/media/RAIDONE/radice/neural-networks-data/predictions'

def crop(img):
    # Perform center cropping, preserving 50% vertically.
    middle_perc = 0.50
    left = 1-middle_perc
    half = left/2
    a = img[int(img.shape[0]*(half)):int(img.shape[0]*(1-half)), :]
    # Resize to match target height while not preserving aspect ratio.
    wdt = WIDTH
    b = cv2.resize(a, (wdt, 128))
    # Perform center cropping horizontally.
    remain = b.shape[1] - 416
    c = b[:, int(remain/2):b.shape[1]-int(remain/2)]

    return c


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
        line = line.rstrip()

        basename = os.path.basename(line).split('.')[0]

        image = cv2.imread(line).astype(np.float32)

        image = crop(image)

        input_batch = np.reshape(image, (1, HEIGHT, WIDTH, 3))

        depth_ax, depth = training_utils.infer(depth_motion_field_model.input_fn_infer(input_image=input_batch),
                                     depth_motion_field_model.loss_fn,
                                     depth_motion_field_model.get_vars_to_restore_fn)

        examples_save_path = os.path.join(OUTPUT_DIR, 'dml-{}'.format(model))

        if not os.path.isdir(examples_save_path):
            os.makedirs(examples_save_path)

        print(examples_save_path)
        # save depth .png
        depth_ax.figure.savefig(os.path.join(examples_save_path, '{}_depth.png'.format(basename)))
        # save depth .npy
        np.save(os.path.join(examples_save_path, '{}_depth.npy'.format(basename)), depth)


if __name__ == '__main__':
  app.run(main)
