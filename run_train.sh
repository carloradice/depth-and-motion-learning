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

# Kitti train file is: "/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/splits/kitti/eigen_train_files.txt"
# Convenzione nome:
# kitti-nome_modello
# Oxford train files are in: "/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/splits/oxford/"
# Convenzione nome:
# oxford-nome_modello

python -m depth_and_motion_learning.depth_motion_field_train \
  --model_dir=/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/models/kitti-1024x320-score80 \
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
    },
    "image_preprocessing": {
      "image_height": 320,
      "image_width": 1024
    }
  }'
