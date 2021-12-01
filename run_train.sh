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

python -m depth_and_motion_learning.depth_motion_field_train \
  --model_dir=/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/models/2014-05-06-14-19 \
  --param_overrides='{
    "model": {
      "input": {
        "data_path": "/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/splits/oxford/combined/2014-05-06-12-54-54_2014-05-14-13-46-12_2014-05-19-12-51-39_train_files.txt"
      }
    },
    "trainer": {
      "init_ckpt": "/media/RAIDONE/radice/neural-networks-data/depth-and-motion-learning/models/resnet18/model.ckpt",
      "init_ckpt_type": "imagenet",
      "max_steps": 125001
    }
  }'
