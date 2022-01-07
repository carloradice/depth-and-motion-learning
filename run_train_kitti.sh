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
# Convention model name
# kitti-'mask-score'-classes-'dim'

python -m depth_and_motion_learning.depth_motion_field_train \
  --model_dir=/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/models/kitti-80-classes-416x128 \
  --param_overrides='{
    "model": {
      "input": {
        "data_path": "/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/splits/kitti/eigen_train_files.txt"
      }
    },
    "trainer": {
      "init_ckpt": "/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/models/resnet18/model.ckpt",
      "init_ckpt_type": "imagenet",
      "max_steps": 200000
    }
  }'
