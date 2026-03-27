# Oral Cancer Detection - Hybrid Model Experiments

This folder contains 10 experiments based on the Hybrid ViT + DenseNet model.
Each file is a standalone training script.

## Experiments List

| File Name | Description | Key Changes |
|-----------|-------------|-------------|
| `exp1_baseline_concat_train.py` | **Baseline** | Concatenation fusion, AdamW, LR 1e-4, Batch 4 |
| `exp2_elementwise_sum_train.py` | **Fusion** | Element-wise Sum (Projected DenseNet + ViT) |
| `exp3_trainable_fusion_train.py` | **Fusion** | Gated Fusion (Learnable weights) |
| `exp4_sgd_optimizer_train.py` | **Optimizer** | SGD (LR 1e-3, Momentum 0.9) |
| `exp5_batch_size_8_train.py` | **Hyperparam** | Batch Size 8 |
| `exp6_lr_1e5_train.py` | **Hyperparam** | Lower Learning Rate (1e-5) |
| `exp7_deep_classifier_train.py` | **Arch** | Deeper Logic (2 Hidden Layers in Classifier) |
| `exp8_reduced_filters_train.py` | **Arch** | Smaller Logic (Hidden Dim 256) |
| `exp9_weighted_sum_train.py` | **Fusion** | Weighted Sum (Learnable scalar alpha) |
| `exp10_adam_train.py` | **Optimizer** | Standard Adam Optimizer |

## Usage

Run any experiment using python:

```bash
python exp1_baseline_concat_train.py
```

Models will be saved in this `experiment` folder with unique names.
