# MNIST MLP Homework

This repository contains the implementation of a configurable Multi-Layer Perceptron (MLP) for handwritten digit classification on the MNIST dataset. The project was developed for a deep learning homework assignment and includes model implementation, training pipeline, ablation studies, and experiment outputs.

## Task
The goal of this homework is to classify digit images into one of the 10 classes (0–9) using an MLP model.

## Project Structure

```text
mnist-mlp-hw/
├── checkpoints/          # Saved model checkpoints
├── data/                 # MNIST dataset files
├── models/               # Model definitions
│   ├── __init__.py
│   └── mlp.py
├── report/               # Experiment outputs and plots
├── main.py               # Main training/evaluation script
├── train.py              # Training, evaluation, and plotting utilities
├── test.py               # Final test evaluation
├── parameters.py         # Configuration dataclass
├── requirements.txt      # Python dependencies
└── README.md
```

## Features
- Configurable MLP architecture
- Variable number of hidden layers
- Variable hidden layer width
- ReLU and GELU activations
- Optional Batch Normalization
- Optional Dropout
- L1 and L2 regularization
- Learning rate scheduler
- Early stopping
- Training/validation loss and accuracy plots

## Installation

Create and activate a virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

A basic example:

```bash
python main.py
```

Example with a custom configuration:

```bash
python main.py --activation gelu --hidden_dims 256 --dropout 0.2 --use_batchnorm true --learning_rate 0.001
```

## Experiment Outputs
Detailed experiment results and generated plots are available in the `report/` directory.  
Saved model checkpoints are stored in the `checkpoints/` directory.

## Best Final Configuration
- Activation: GELU
- Hidden layers: 1
- Hidden dimension: 256
- Dropout: 0.2
- Batch Normalization: True
- Learning rate: 0.001
- L1 regularization: 0.00001
- L2 regularization: 0.0
- Early stopping: triggered at epoch 17
- Best validation loss: 0.0634
- Final test accuracy: 0.9821

## Notes
This repository includes the implementation code, experiment outputs for the homework.