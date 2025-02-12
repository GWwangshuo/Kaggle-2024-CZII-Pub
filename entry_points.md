# Entry Points

This document outlines the primary entry points of the project, detailing their respective functions. For comprehensive usage instructions, please refer to the "Usage" section in the [README.md](./README.md).

## 1. Data Preparation
- **Description**: This script downloads the raw training data, generates segmentation masks, and outputs the corresponding mask labels.
- **Inputs**: 
  - `RAW_DATA_DIR` (as specified in `SETTINGS.json`)
- **Outputs**:
  - `RAW_DATA_DIR/working` (as specified in `SETTINGS.json`)
- **Detailed Instructions**: Refer to the "Data Preparation and Label Generation" subsection in the "Usage" section of the README.md.

## 2. Model Training
- **Description**: This script facilitates the training of the model using the prepared training data.
- **Inputs**:
  - `TRAIN_DATA_WORKING_DIR` (as specified in `SETTINGS.json`)
- **Outputs**:
  - `MODEL_CHECKPOINT_DIR` (as specified in `SETTINGS.json`)
- **Detailed Instructions**: Refer to the "Model Training" subsection in the "Usage" section of the [README.md](./README.md).

## 3. Model Prediction
- **Description**: This script loads a pre-trained model and generates predictions on new data samples.
- **Inputs**:
  - `RAW_DATA_DIR` (as specified in `SETTINGS.json`)
  - `MODEL_CHECKPOINT_DIR` (as specified in `SETTINGS.json`)
- **Outputs**:
  - `submission.csv`
- **Detailed Instructions**: Refer to the "Model Prediction" subsection in the "Usage" section of the [README.md](./README.md).
