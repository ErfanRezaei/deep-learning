# HW2: Data Augmentation and Adversarial Samples

This repository contains my implementation for Homework 2 of the Deep Learning course.

## Overview

In this homework, I evaluated the robustness of deep neural networks on CIFAR-10 under:
- clean samples
- corrupted samples from CIFAR-10-C
- adversarial attacks with PGD20
- AugMix-based training
- knowledge distillation
- adversarial transferability

The teacher model is based on ResNet-18, and the student model is SimpleCNN.

## Repository Structure

- `main.py`: main entry point
- `train.py`: training pipeline
- `test.py`: evaluation pipeline
- `parameters.py`: experiment parameters
- `compute_flops.py`: FLOPs computation
- `models/`: model definitions
- `utils/`: attacks, augmentations, Grad-CAM, t-SNE, metrics, and data utilities
- `results/figures/`: saved visualizations and outputs

## Implemented Experiments

- Clean accuracy evaluation on CIFAR-10
- Corruption robustness evaluation on CIFAR-10-C
- AugMix-based fine-tuning
- PGD20 adversarial attacks with `L∞ (ε = 4/255)` and `L2 (ε = 0.25)`
- Grad-CAM visualization on clean and adversarial samples
- t-SNE visualization of clean and adversarial features
- Knowledge distillation with baseline and AugMix-trained teachers
- Transferability analysis of teacher-generated adversarial examples

## Main Findings

- The baseline teacher achieved better clean and corruption performance.
- The AugMix teacher showed a small advantage under adversarial attacks.
- The baseline teacher produced the better distilled student in this setup.
- Adversarial examples generated on the teacher transferred to the student to a limited extent.

## Requirements

Install dependencies with:

`pip install -r requirements.txt`

## Run

Main script:

`python main.py`

Evaluation:

`python test.py`

FLOPs:

`python compute_flops.py`




