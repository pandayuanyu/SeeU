#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


evalset=(
blackswan
drone_dollyin2
)

DATA_PATH=/home/images/


CKPT_PATH=checkpoints/megasam_final.pth




for seq in ${evalset[@]}; do
    CUDA_VISIBLE_DEVICE=0 python camera_tracking_scripts/test_demo.py \
    --datapath=$DATA_PATH/$seq \
    --weights=$CKPT_PATH \
    --scene_name $seq \
    --mono_depth_path $(pwd)/Depth-Anything/video_visualization \
    --metric_depth_path $(pwd)/UniDepth/outputs \
    --disable_vis $@ 

done

