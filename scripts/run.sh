#!/bin/bash

export PYTHONPATH=./

set -e

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "@        Download and Preprocess the Data          @"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
python3 src/utils/prepare_data.py
python3 src/utils/generate_segmentation_mask.py


echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "@           Train Segmentation Models              @"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

configs=(
    "monai_unet_v1.yaml"
    "monai_unet_v2.yaml"
    "monai_unet_v3.yaml"
    "monai_unet_v4.yaml"
    "voxhrnet.yaml"
    "voxresnet.yaml"
    "unet2e3d.yaml"
)

valid_ids=(
    "TS_6_4" 
    "TS_5_4" 
    "TS_69_2" 
    "TS_6_6" 
    "TS_73_6" 
    "TS_86_3" 
    "TS_99_9"
)

for vid in "${valid_ids[@]}"; do
    for config in "${configs[@]}"; do
        python src/train.py --config "src/config/$config" --valid_id "$vid"
    done
done
