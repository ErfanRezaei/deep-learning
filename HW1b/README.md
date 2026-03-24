# HW1b - Transfer learning and knowledge distillation

This folder contains the implementation of the second deep learning homework, which focuses on **Transfer Learning** and **Knowledge Distillation** on the **CIFAR-10** dataset.

The project is implemented in **PyTorch** and organized in a modular structure for training, testing, model definition, and complexity analysis.

---

## Project Structure

```text
HW1b/
├── models/
│   ├── __init__.py
│   ├── mobilenet_student.py
│   ├── resnet_cifar.py
│   ├── simple_cnn.py
│   └── transfer_learning.py
├── utils/
│   ├── data.py
│   ├── flops.py
│   ├── losses.py
│   └── metrics.py
├── compute_flops.py
├── main.py
├── parameters.py
├── test.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses the **CIFAR-10** dataset.

The dataset is downloaded automatically when needed, so the `data/` folder is not included in the repository.

---

## Notes

Due to GitHub file size limits:
- `data/` is not included
- `checkpoints/` is not included

---

## Run Experiments

### Sanity Check

```bash
python main.py --mode sanity_check --task simple_cnn --model-name simple_cnn --experiment-name simple_cnn_model_check
```

### SimpleCNN Baseline

```bash
python main.py --mode train --task simple_cnn --model-name simple_cnn --experiment-name simple_cnn_baseline --epochs 10 --optimizer-name adam --learning-rate 0.001
```

### ResNet (No Label Smoothing)

```bash
python main.py --mode train --task resnet_scratch --model-name resnet_cifar --experiment-name resnet_no_ls --epochs 10 --optimizer-name adam --learning-rate 0.001
```

### ResNet (With Label Smoothing)

```bash
python main.py --mode train --task resnet_scratch --model-name resnet_cifar --experiment-name resnet_ls --epochs 10 --optimizer-name adam --learning-rate 0.001 --label-smoothing 0.1
```

### SimpleCNN + Knowledge Distillation

```bash
python main.py --mode train --task distill_simple_cnn --model-name simple_cnn --teacher-model-name resnet_cifar --teacher-checkpoint checkpoints/resnet_no_ls_best.pt --use-distillation --experiment-name simple_cnn_kd --epochs 10 --optimizer-name adam --learning-rate 0.001 --alpha 0.7 --temperature 4.0
```

### MobileNet Baseline

```bash
python main.py --mode train --task mobilenet_student --model-name mobilenet_student --experiment-name mobilenet_student_baseline --epochs 8 --optimizer-name adam --learning-rate 0.001
```

---

## Transfer Learning

### Method 1: Pretrained + Resize + Freeze

```bash
python main.py --mode train --task transfer_learning --model-name transfer_resnet18_resize --experiment-name transfer_resize_freeze --epochs 3 --optimizer-name adam --learning-rate 0.001 --use-pretrained --freeze-early-layers --use-imagenet-size --resize-to 224 --train-batch-size 16 --eval-batch-size 32 --num-workers 0
```

### Method 2: Pretrained + Modified Stem + Fine-tune

```bash
python main.py --mode train --task transfer_learning --model-name transfer_resnet18_cifar --experiment-name transfer_cifar_finetune --epochs 6 --optimizer-name adam --learning-rate 0.001 --use-pretrained
```

---

## Compute FLOPs / MACs

### SimpleCNN

```bash
python compute_flops.py --model-name simple_cnn --image-size 32
```

### ResNet

```bash
python compute_flops.py --model-name resnet_cifar --image-size 32
```

### MobileNet

```bash
python compute_flops.py --model-name mobilenet_student --image-size 32
```

---

## Main Results

- Best overall model: **ResNet without label smoothing**
- Best teacher model: **ResNet without label smoothing**
- Knowledge distillation improved **SimpleCNN** without changing inference complexity
- **MobileNet** was the most efficient model in terms of MACs, but it achieved lower accuracy than the other completed models

---

## Author

Erfan Rezaei
