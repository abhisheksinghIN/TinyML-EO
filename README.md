# TinyML-EO

A lightweight semantic segmentation framework for **Earth Observation (EO)** imagery using quantized deep learning models.

The project implements a U-Net-based segmentation model for **Sentinel-1 (SAR)** and **Sentinel-2 (multispectral)** data, with support for INT8 quantization and ONNX export for edge/TinyML deployment.

## Features

- U-Net semantic segmentation
- INT8 quantized U-Net using [Brevitas](https://github.com/Xilinx/brevitas)
- Sentinel-1, Sentinel-2, and S1+S2 inputs
- 256 × 256 image chips
- 5-class segmentation
- PyTorch, ONNX Runtime, and OpenVINO inference
- Accuracy and inference-performance evaluation

## Dataset

The repository currently contains CSV split files for the DFC2020 dataset:

```text
dataset/CSV_list/
├── C_NYC-train_split.csv
├── C_NYC-val_split.csv
└── C_NYC-test.csv
```

## Input Mode

| Mode     | Channels |
| -------- | -------: |
| `s1only` |        2 |
| `s2only` |       13 |
| `s1s2`   |       15 |

## Repository Structure

```text
TinyML-EO/
├── dataset/CSV_list/     # Dataset split files
├── networks/soaint.py    # U-Net and quantized U-Net
├── main_INT8.py          # Training & ONNX export
├── evaluate_INT8.py      # Evaluation & benchmarking
└── utilsint.py           # Dataset utilities
```

## Installation

```bash
pip install torch torchvision
pip install numpy pandas rasterio tqdm scikit-learn
pip install brevitas onnx onnxruntime openvino
```

## Training

Train the quantized model with Sentinel-1:

```bash
python main_INT8.py --model qnn --s_mode s1only
```

Sentinel-2:

```bash
python main_INT8.py --model qnn --s_mode s2only
```

Sentinel-1 + Sentinel-2:

```bash
python main_INT8.py --model qnn --s_mode s1s2
```

The training script performs validation, saves checkpoints, folds BatchNorm layers, and exports the quantized model to **ONNX QCDQ format** and further exported to OpenVINO.

## Evaluation

Evaluate a PyTorch checkpoint:

```bash
python evaluate_INT8.py \
    --model_path ./log_INT8/qnn_s1/qnn_epoch_30.pth \
    --model qnn \
    --s_mode s1only
```

Evaluate an OpenVINO model:

```bash
python evaluate_INT8.py \
    --model_path ./log_INT8/qnn_s1/openvino_qnn.xml \
    --model qnn \
    --s_mode s1only
```

## Evaluation Metrics

The evaluation pipeline reports:

- Overall Accuracy (OA)
- Mean IoU (mIoU)
- Cohen's Kappa
- Confusion Matrix
- Classification Report
- Inference latency and throughput
- CPU/GPU memory usage

## Model Deployment

The quantized model is exported to **ONNX QCDQ** after BatchNorm folding, providing a deployment-friendly representation for further optimization and edge workflows.

