# Quantized Deep Learning Model for Earth Observation Application

A lightweight semantic segmentation framework for Earth Observation (EO) imagery using quantized deep learning models.

The project implements a Quantized U-Net-based segmentation model for Sentinel-1 (SAR) and Sentinel-2 (multispectral) data, with support for INT8 quantization and ONNX export for edge/TinyML deployment.

# Features

U-Net semantic segmentation

INT8 quantized U-Net using Brevitas

Sentinel-1, Sentinel-2, and S1+S2 inputs

256 × 256 image chips

PyTorch, ONNX Runtime, and OpenVINO inference


# Input Mode

Mode	Channels

s1only	2

s2only	13

s1s2	15

# Repository Structure

TinyML-EO/

├── dataset/CSV_list/     # Dataset split files

├── networks/soaint.py    # U-Net and quantized U-Net

├── main_INT8.py          # Training & ONNX export

├── evaluate_INT8.py      # Evaluation & benchmarking

└── utilsint.py           # Dataset utilities


# Training
Train the quantized model with Sentinel-1:
python main_INT8.py --model qnn --s_mode s1only

Sentinel-2:
python main_INT8.py --model qnn --s_mode s2only

Sentinel-1 + Sentinel-2:
python main_INT8.py --model qnn --s_mode s1s2

The training script performs validation, saves checkpoints, folds BatchNorm layers, and exports the quantized model to ONNX QCDQ format.
