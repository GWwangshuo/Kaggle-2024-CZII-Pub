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

#### 2. Data Preparation
Ensure the raw dataset is stored in `<RAW_DATA_DIR>`, then run:
```bash
python3 src/utils/prepare_data.py
python3 src/utils/generate_segmentation_mask.py
```

#### 3. Model Training
Train particle segmentation models using different configurations:

- **MONAI U-Net (1 residual unit, validation per tomogram, spatial size: 128x256x256)**
  ```bash
  python src/train.py --config src/config/monai_unet_v1.yaml --valid_id TS_6_4
  python src/train.py --config src/config/monai_unet_v1.yaml --valid_id TS_5_4
  ```
  *(Repeat for additional tomograms as needed)*

- **MONAI U-Net (2 residual units, dropout 0.3, validation per tomogram, spatial size: 128x256x256)**
  ```bash
  python src/train.py --config src/config/monai_unet_v2.yaml --valid_id TS_6_4
  python src/train.py --config src/config/monai_unet_v2.yaml --valid_id TS_5_4
  ```
  *(Repeat for additional tomograms as needed)*

- **Additional Models (DenseVNet, VoxHRNet, VoxResNet)**
  ```bash
  python src/train.py --config src/config/densevnet.yaml --valid_id TS_6_4
  python src/train.py --config src/config/voxhrnet.yaml --valid_id TS_6_4
  python src/train.py --config src/config/voxresnet.yaml --valid_id TS_6_4
  ```
  *(Repeat for additional tomograms as needed, and refer to respective configuration files for tuning.)*

## License

This project is licensed under the MIT License.

