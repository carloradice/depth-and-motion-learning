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

#!/bin/bash
set -e
set -x

# Kitti train file is in: "/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/splits/kitti/"
# Choose between available train files (different 'mask-score' and 'dim')
# eigen_train_files_80-classes-416x128.txt
# eigen_train_files_80-classes-640x192.txt
# Convention model name
# kitti-'mask-score'-classes-'dim'
# "image_height": 128/192
# "image_width": 416/640

python -m depth_and_motion_learning.depth_motion_field_train \
  --model_dir=/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/models/oxford-416x128\
  --param_overrides='{
    "model": {
      "input": {
        "data_path": "/media/RAIDONE/radice/neural-networks-data/struct2depth/splits/oxford_train_files.txt",
      },
      "image_preprocessing": {
        "data_augmentation": True,
        "image_height": 128,
        "image_width": 416,
      },
    },
    "trainer": {
      "init_ckpt": "/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/models/resnet18/model.ckpt",
      "init_ckpt_type": "imagenet",
      "max_steps": 200000,
    },
    "learn_egomotion": False,
  }'
