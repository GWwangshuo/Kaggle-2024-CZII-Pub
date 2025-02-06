# CZII - CryoET Object Identification - 2nd Place Solution

This is the 2nd place solution for the CZII - CryoET Object Identification competition. For more details, please refer to the [discussion](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/561568).

## Environment

This project was developed in the following environment:

### Workstation

- **CPU**: Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz
- **Memory**: 256 GB RAM
- **GPU**: NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **OS**: Ubuntu 20.04.6 (5.4.0-181-generic)


## Prerequisites

Before running this project, make sure the following are already installed on your system:

- **[NVIDIA Driver](https://www.nvidia.com/en-us/drivers/)**
- **[CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)**
- **[Docker](https://docs.docker.com/engine/install/debian/)**  
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**  
- **[Kaggle API](https://www.kaggle.com/docs/api)**

The Kaggle API must be set up with your API token located at `~/.kaggle/kaggle.json`. Refer to Kaggle API documentation for instructions on generating your API token.

## Setup

This project is set up to run inside a Docker container.

1. Clone the Repository

   ```bash
   git clone https://github.com/GWwangshuo/Kaggle-2024-CZII-Pub.git
   ```

2. Move into the Project Directory

   ```bash
   cd Kaggle-2024-CZII-Pub
   ```

3. Build the Docker Image

   ```bash
   docker build -t czii2024_2nd_img .
   ```

4. Run the Docker Container

   ```bash
   docker run --gpus all -it --rm --name czii2024_2nd_cont --shm-size 24G -v $(pwd):/kaggle -v ~/.kaggle:/root/.kaggle czii2024_2nd_img /bin/bash
   ```

## Usage

To run the entire workflow, including data preparation, model training, and prediction, use the following script:

```bash
scripts/run.sh
```

This script sequentially performs all necessary steps to complete the workflow.

#### Step by Step

If you prefer to run each step individually, follow the instructions below.

1. Data Download and Preprocessing

   Prepare the dataset in the `<RAW_DATA_DIR>` directory and run:

   ```bash
   python3 src/utils/prepare_data.py
   python3 src/utils/generate_segmentation_mask.py
   ```

2. Train Particle Segmentation Models

   Start by creating the datasets and training the segmentation models.

   ```bash

    # monai unet with 1 resunit, validate on each tomogram, spatial size (128, 256, 256)
    python src/train.py --config src/config/monai_unet_v1.yaml --valid_id TS_6_4
    python src/train.py --config src/config/monai_unet_v1.yaml --valid_id TS_5_4
    python src/train.py --config src/config/monai_unet_v1.yaml --valid_id TS_69_2
    python src/train.py --config src/config/monai_unet_v1.yaml --valid_id TS_6_6
    python src/train.py --config src/config/monai_unet_v1.yaml --valid_id TS_73_6
    python src/train.py --config src/config/monai_unet_v1.yaml --valid_id TS_86_3
    python src/train.py --config src/config/monai_unet_v1.yaml --valid_id TS_99_9

    # monai unet with 2 resunit, dropout 0.3, validate on each tomogram, spatial size (128, 256, 256)
    python src/train.py --config src/config/monai_unet_v2.yaml --valid_id TS_6_4
    python src/train.py --config src/config/monai_unet_v2.yaml --valid_id TS_5_4
    python src/train.py --config src/config/monai_unet_v2.yaml --valid_id TS_69_2
    python src/train.py --config src/config/monai_unet_v2.yaml --valid_id TS_6_6
    python src/train.py --config src/config/monai_unet_v2.yaml --valid_id TS_73_6
    python src/train.py --config src/config/monai_unet_v2.yaml --valid_id TS_86_3
    python src/train.py --config src/config/monai_unet_v2.yaml --valid_id TS_99_9

    # monai unet with 1 resunit, validate on each tomogram, spatial size (128, 384, 384)
    python src/train.py --config src/config/monai_unet_v3.yaml --valid_id TS_6_4
    python src/train.py --config src/config/monai_unet_v3.yaml --valid_id TS_5_4
    python src/train.py --config src/config/monai_unet_v3.yaml --valid_id TS_69_2
    python src/train.py --config src/config/monai_unet_v3.yaml --valid_id TS_6_6
    python src/train.py --config src/config/monai_unet_v3.yaml --valid_id TS_73_6
    python src/train.py --config src/config/monai_unet_v3.yaml --valid_id TS_86_3
    python src/train.py --config src/config/monai_unet_v3.yaml --valid_id TS_99_9

    # monai unet with 2 resunit, dropout 0.3, validate on each tomogram, spatial size (128, 384, 384)
    python src/train.py --config src/config/monai_unet_v4.yaml --valid_id TS_6_4
    python src/train.py --config src/config/monai_unet_v4.yaml --valid_id TS_5_4
    python src/train.py --config src/config/monai_unet_v4.yaml --valid_id TS_69_2
    python src/train.py --config src/config/monai_unet_v4.yaml --valid_id TS_6_6
    python src/train.py --config src/config/monai_unet_v4.yaml --valid_id TS_73_6
    python src/train.py --config src/config/monai_unet_v4.yaml --valid_id TS_86_3
    python src/train.py --config src/config/monai_unet_v4.yaml --valid_id TS_99_9

    # densevnet, dropout 0.3, validate on each tomogram, spatial size (128, 256, 256)
    python src/train.py --config src/config/densevnet.yaml --valid_id TS_6_4
    python src/train.py --config src/config/densevnet.yaml --valid_id TS_5_4
    python src/train.py --config src/config/densevnet.yaml --valid_id TS_69_2
    python src/train.py --config src/config/densevnet.yaml --valid_id TS_6_6
    python src/train.py --config src/config/densevnet.yaml --valid_id TS_73_6
    python src/train.py --config src/config/densevnet.yaml --valid_id TS_86_3
    python src/train.py --config src/config/densevnet.yaml --valid_id TS_99_9

    # voxhrnet, dropout 0.1, validate on each tomogram, spatial size (128, 256, 256)
    python src/train.py --config src/config/voxhrnet.yaml --valid_id TS_6_4
    python src/train.py --config src/config/voxhrnet.yaml --valid_id TS_5_4
    python src/train.py --config src/config/voxhrnet.yaml --valid_id TS_69_2
    python src/train.py --config src/config/voxhrnet.yaml --valid_id TS_6_6
    python src/train.py --config src/config/voxhrnet.yaml --valid_id TS_73_6
    python src/train.py --config src/config/voxhrnet.yaml --valid_id TS_86_3
    python src/train.py --config src/config/voxhrnet.yaml --valid_id TS_99_9

    # voxresnet, validate on each tomogram, spatial size (128, 256, 256)
    python src/train.py --config src/config/voxresnet.yaml --valid_id TS_6_4
    python src/train.py --config src/config/voxresnet.yaml --valid_id TS_5_4
    python src/train.py --config src/config/voxresnet.yaml --valid_id TS_69_2
    python src/train.py --config src/config/voxresnet.yaml --valid_id TS_6_6
    python src/train.py --config src/config/voxresnet.yaml --valid_id TS_73_6
    python src/train.py --config src/config/voxresnet.yaml --valid_id TS_86_3
    python src/train.py --config src/config/voxresnet.yaml --valid_id TS_99_9
   ```


## License

This project is licensed under the MIT License.
