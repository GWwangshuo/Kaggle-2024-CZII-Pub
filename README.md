# CZII - CryoET Object Identification - 2nd Place Solution

This repository contains the 2nd place solution for the CZII - CryoET Object Identification competition. For a detailed discussion, please refer to the [competition discussion thread](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/561568). Some components of this codebase are derived from [this source](https://github.com/Moyasii/Kaggle-2024-RSNA-Pub).

## Environment

This project was developed using the following hardware and software environment:

### Workstation Specifications

- **CPU**: Intel(R) Xeon(R) Gold 5218R @ 2.10GHz
- **Memory**: 256 GB RAM
- **GPU**: NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **Operating System**: Ubuntu 20.04.6 (Kernel 5.4.0-181-generic)

## Prerequisites

Ensure the following dependencies are installed before running the project:

- **[NVIDIA Driver](https://www.nvidia.com/en-us/drivers/)**
- **[CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)**
- **[Docker](https://docs.docker.com/engine/install/debian/)**
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**
- **[Kaggle API](https://www.kaggle.com/docs/api)**

To use the Kaggle API, ensure your API token is configured in `~/.kaggle/kaggle.json`. Refer to the [Kaggle API documentation](https://www.kaggle.com/docs/api) for instructions on generating and setting up your API token.

## Setup

This project is designed to run within a Docker container.

1. Clone the Repository:
   ```bash
   git clone https://github.com/GWwangshuo/Kaggle-2024-CZII-Pub.git
   ```
2. Navigate to the Project Directory:
   ```bash
   cd Kaggle-2024-CZII-Pub
   ```
3. Build the Docker Image:
   ```bash
   docker build -t czii2024_2nd_img .
   ```
4. Run the Docker Container:
   ```bash
   docker run --gpus all -it --rm --name czii2024_2nd_cont --shm-size 24G -v $(pwd):/kaggle -v ~/.kaggle:/root/.kaggle czii2024_2nd_img /bin/bash
   ```

## Usage

To execute the complete workflow, including data preparation, model training, and inference, run:
```bash
bash scripts/run.sh
```

### Step-by-Step Execution

For a more granular approach, execute each step as follows:

#### 1. Set Environment Variable
```bash
export PYTHONPATH=./
```

#### 2. Data Preparation and Label Generation
Ensure the raw dataset is stored in `<RAW_DATA_DIR>`, then run:
```bash
python3 src/utils/prepare_data.py
python3 src/utils/generate_segmentation_mask.py
```

#### 3. Model Training
Train particle segmentation models using different configurations:

- **MONAI U-Net (1 residual unit, spatial size: 128x256x256, 7 classes)**
  ```bash
  python src/train.py --config src/config/monai_unet_v1.yaml --valid_id TS_6_4
  python src/train.py --config src/config/monai_unet_v1.yaml --valid_id TS_5_4
  ```
  *(Repeat for additional tomograms as needed)*

- **MONAI U-Net (2 residual units, dropout 0.3, spatial size: 128x256x256, 7 classes)**
  ```bash
  python src/train.py --config src/config/monai_unet_v2.yaml --valid_id TS_6_4
  python src/train.py --config src/config/monai_unet_v2.yaml --valid_id TS_5_4
  ```
  *(Repeat for additional tomograms as needed)*

- **MONAI U-Net (1 residual unit, spatial size: 128x256x256, 6 classes)**
  ```bash
  python src/train.py --config src/config/monai_unet_v3.yaml --valid_id TS_6_4
  python src/train.py --config src/config/monai_unet_v3.yaml --valid_id TS_5_4
  ```
  *(Repeat for additional tomograms as needed)*

- **Additional Models (DenseVNet, VoxHRNet, VoxResNet, SegResNet, UNet2E3D)**
  ```bash
  python src/train.py --config src/config/densevnet.yaml --valid_id TS_6_4
  python src/train.py --config src/config/voxhrnet.yaml --valid_id TS_6_4
  python src/train.py --config src/config/voxresnet.yaml --valid_id TS_6_4
  python src/train.py --config src/config/segresnet.yaml --valid_id TS_6_4
  python src/train.py --config src/config/unet2e3d.yaml --valid_id TS_6_4
   ```
  *(Repeat for additional tomograms as needed, and refer to respective configuration files for tuning.)*


#### 4. Model Prediction
1. Download the pretrained checkpoints from [this link](https://www.kaggle.com/datasets/sjtuwangshuo/czii2024-best-ckpts).
2. Update the dataset path `DATA_KAGGLE_DIR` and the absolute checkpoint path `best_weights_root` in the `submit.ipynb` notebook.
3. Excute the `submit.ipynb` notebook.

##### Model Performance with 7 TTA Summary

| **No** | **Model**                                                |  **Developer**  | **Architecture** | **Parameters** |    **Valid ID**   | **Normalization** | **Activation** | **Public LB** | **Private LB** |
| ------ | -------------------------------------------------------- | ----------------| ---------------- | -------------- | ----------------- | ----------------- | -------------- | ------------- | -------------- |
| 1      | epoch122-step2952-valid_loss0.3625-val_metric0.8367.ckpt | Lion            | UNet3D           | 1.1M           | TS_86_3           | InstanceNorm3d    | PReLU          | 0.77379       | 0.76582        |
| 2      | epoch148-step3576-valid_loss1.1154-val_metric0.7722.ckpt | Lion            | UNet3D           | 1.1M           | TS_6_4            | InstanceNorm3d    | PReLU          | 0.77021       | 0.76725        |
| 3      | epoch153-step3696-valid_loss0.3021-val_metric0.8900.ckpt | Lion            | UNet3D           | 1.6M           | TS_69_2           | InstanceNorm3d    | PReLU          | 0.77205       | 0.76676        |
| 4      | epoch194-step4680-valid_loss1.0213-val_metric0.8788.ckpt | Lion            | UNet3D           | 1.1M           | TS_69_2           | InstanceNorm3d    | PReLU          | 0.77390       | 0.76737        |
| 5      | epoch138-step3336-valid_loss0.3690-val_metric0.8476.ckpt | Lion            | UNet3D           | 1.1M           | TS_73_6           | InstanceNorm3d    | PReLU          | 0.76543       | 0.76025        |
| 6      | epoch152-step3672-valid_loss0.4333-val_metric0.7929.ckpt | Lion            | DenseVNet        | 873K           | TS_6_6            | InstanceNorm3d    | PReLU          | 0.76528       | 0.75417        |
| 7      | epoch195-step4704-valid_loss0.4258-val_metric0.7914.ckpt | Lion            | VoxResNet        | 7.0M           | TS_6_6            | InstanceNorm3d    | PReLU          | 0.77457       | 0.76593        |
| 8      | epoch188-step4536-valid_loss0.4231-val_metric0.8659.ckpt | Lion            | VoxHRNet         | 1.4M           | TS_73_6           | InstanceNorm3d    | PReLU          | 0.76738       | 0.75995        |
| 9      | epoch198-step4776-valid_loss0.3471-val_metric0.8730.ckpt | Lion            | VoxHRNet         | 1.4M           | TS_73_6           | InstanceNorm3d    | PReLU          | 0.76135       | 0.75848        |
| 10     | epoch133-val_loss0.52-val_metric0.56-step3216.ckpt       | Luoziqian       | UNet3D           | 1.1M           | TS_6_4            | BatchNorm3d       | PReLU          | 0.76844       | 0.76320        |
| 11     | epoch314-val_loss0.54-val_metric0.54-step7560.ckpt       | Luoziqian       | SegResNet        | 1.2M           | TS_6_4            | GroupNorm         | ReLU           | 0.75521       | 0.74647        |
| 12     | epoch114-val_loss0.55-val_metric0.53-Step2760.ckpt       | Luoziqian       | UNet2E3D         | 14.2M          | TS_6_4            | BatchNorm3d       | ReLU           | 0.73758       | 0.72966        |


## License
This project is licensed under the MIT License, see the [LICENSE.txt](./LICENSE.txt) file for details.

