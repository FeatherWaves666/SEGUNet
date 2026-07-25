# SEG-UNet: Spectral–Edge Gated U-Net for Infrared Small-Target Segmentation

Official PyTorch implementation of the paper **"SEG-UNet: Spectral–Edge Gated U-Net for Infrared Small-Target Segmentation"**.

---

## Abstract

Infrared small-target detection (IRSTD) presents critical challenges due to low thermal contrast, complex background clutter, and the minute sub-pixel scale of target signatures. Conventional encoder-decoder architectures struggle in these low-signal thermal regimes due to high-frequency signal loss during spatial downsampling, clutter noise propagation across skip connections, and boundary distortion during upsampling.

To address these structural deficiencies, we propose **SEG-UNet**, a specialized segmentation framework that integrates frequency conservation and geometric gradient priors directly into the dense prediction pipeline:
1. **High-frequency Preserving Downsampler (HPD)**: Employs a dual-branch discrete Haar wavelet transform (DWT) and MaxPool to decompose the feature space, retaining high-frequency spectral components and preventing pooling-induced energy dilution.
2. **Clutter-suppressing Cross-scale Gating (CCG)**: Utilizes deep abstract semantics as statistical anchors to dynamically filter shallow background noise across skip-connections.
3. **Boundary-guided Geometric Restorer (BGR)**: Applies fixed Sobel gradient operators as explicit geometric priors to constrain boundary gradients during upsampling, reconstructing sharp target contours.

---

## Architecture Overview

![SEG-UNet Architecture Overview](assets/overview.png)

*Figure 1: The overall architecture of the proposed SEG-UNet and its specialized structural components: (a) High-frequency Preserving Downsampler (HPD), (b) Macro Backbone, (c) Boundary-guided Geometric Restorer (BGR), (d) Micro Assembly, and (f) Clutter-suppressing Cross-scale Gating (CCG).*

Mathematical formulation of key structural interventions:
- **HPD**: $\text{HPD}(x) = \psi( [ x_{\text{pool}} \odot s_1, x_{\text{dwt}} \odot s_2 ] )$
- **CCG**: $\text{CCG}(x_e, x_d) = [x_e, x_d] \odot s$
- **BGR**: $\text{BGR}(x) = x^{\text{up}} \odot (1 + a)$

---

## Quantitative & Qualitative Results

### Benchmark Quantitative Comparison on IRSTD-1k and NUAA-SIRST

> **Note**: **Bold** indicates the best score, and <u>underline</u> indicates the second-best score.

| Method | Type | IRSTD-1k $IoU$ (%) ↑ | IRSTD-1k $P_d$ (%) ↑ | IRSTD-1k $F_a$ ($\times 10^{-6}$) ↓ | NUAA-SIRST $IoU$ (%) ↑ | NUAA-SIRST $P_d$ (%) ↑ | NUAA-SIRST $F_a$ ($\times 10^{-6}$) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Max-Median** (1999) | Filtering | 7.00 | 65.21 | 59.73 | 4.17 | 69.20 | 55.33 |
| **Top-Hat** (2010) | Filtering | 10.06 | 75.11 | 1432 | 7.14 | 79.84 | 1012 |
| **TLLCM** (2020) | Local Contrast | 3.31 | 77.39 | 6738 | 1.03 | 79.09 | 5899 |
| **WSLCM** (2021) | Local Contrast | 3.45 | 72.44 | 6619 | 1.16 | 77.95 | 5446 |
| **IPI** (2013) | Low Rank | 27.92 | 81.37 | 16.18 | 25.67 | 85.55 | 11.47 |
| **RIPT** (2017) | Low Rank | 14.11 | 77.55 | 28.31 | 11.05 | 79.08 | 22.61 |
| **NRAM** (2018) | Low Rank | 15.25 | 70.68 | 16.93 | 15.25 | 70.68 | 16.93 |
| **PSTNN** (2019) | Low Rank | 24.57 | 71.99 | 35.26 | 30.30 | 72.80 | 48.99 |
| **MSLSTIPT** (2020) | Low Rank | 11.43 | 79.03 | 1524 | 1.08 | 0.05 | 8.18 |
| **MDvsFA** (CVPR 2019) | Deep Learning | 37.34 | 83.71 | 88.52 | 60.30 | 89.35 | 56.35 |
| **ACMNet** (WACV 2021) | Deep Learning | 59.23 | 93.27 | 65.28 | 70.77 | 93.08 | 3.70 |
| **ALCNet** (TGRS 2021) | Deep Learning | 65.68 | 89.25 | 27.71 | 73.74 | 97.25 | 26.79 |
| **ISNet** (CVPR 2022) | Deep Learning | 62.88 | 92.59 | 27.92 | 74.16 | 97.99 | 8.35 |
| **DNANet** (TIP 2022) | Deep Learning | 65.71 | 91.84 | 17.61 | 74.31 | 98.17 | 15.97 |
| **UIUNet** (TIP 2023) | Deep Learning | 65.06 | 91.16 | 12.68 | 72.69 | <u>99.08</u> | 26.61 |
| **IRPruneDet** (AAAI 2024) | Deep Learning | 64.54 | 91.74 | 16.04 | 75.12 | 98.61 | 2.96 |
| **MSHNet** (CVPR 2024) | Deep Learning | 67.16 | <u>93.88</u> | 15.03 | 74.60 | <u>99.08</u> | 17.21 |
| **IRSAM** (ECCV 2024) | Deep Learning | 64.65 | 90.57 | 16.61 | 71.44 | 92.66 | 7.53 |
| **HCFNet** (ICME 2024) | Deep Learning | 64.26 | 92.86 | 23.91 | 72.74 | 98.17 | 6.21 |
| **SCTransNet** (TGRS 2024) | Deep Learning | 68.64 | 91.84 | 11.92 | 77.09 | 98.17 | 15.26 |
| **PConv** (AAAI 2025) | Deep Learning | 67.08 | 92.18 | 11.92 | 76.25 | <u>99.08</u> | 6.74 |
| **SFCANet** (TAES 2025) | Deep Learning | 66.68 | 92.89 | 12.69 | 78.46 | 97.24 | 8.02 |
| **NS-FPN** (CVPR 2026) | Deep Learning | <u>69.29</u> | **95.24** | <u>8.58</u> | **78.75** | **100.0** | <u>1.60</u> |
| **SEG-UNet (Ours)** | Frequency-Guided | **70.14** | <u>93.88</u> | **4.71** | <u>78.57</u> | **100.0** | **1.06** |

---

### Model Complexity, Speed, and Accuracy (on IRSTD-1k)

| Method | Parameters (M) ↓ | FLOPs (G) ↓ | FPS ↑ | IoU (%) ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **MSHNet** | 4.07 | 6.11 | 19.70 | 67.16 |
| **PConv** | 4.06 | 6.03 | 18.54 | 67.08 |
| **NS-FPN** | 4.17 | 7.97 | 14.83 | 69.29 |
| **SCTransNet** | 11.33 | 10.12 | 13.53 | 68.64 |
| **SEG-UNet (Ours)** | **4.92** | **7.00** | **16.13** | **70.14** |

---

### Qualitative Comparison across Complex Scenarios

![Qualitative Visual Results](assets/visual_result.png)

*Figure 2: Qualitative comparison of SEG-UNet against state-of-the-art methods across 5 challenging IRSTD scenarios (Water Clutter, Complex Background, Road, Sky Clutter, and Terrain Clutter). Red, yellow, and green boxes denote false alarms, missed detections, and correct detections, respectively.*

---

## Repository Structure

```text
SEG-UNet/
├── assets/
│   ├── overview.png        # Architecture Overview Diagram (Figure 1)
│   └── visual_result.png   # Qualitative Visual Comparison (Figure 2)
├── model/
│   ├── __init__.py
│   ├── SEG_UNet.py         # Main SEG-UNet model (HPD, CCG, BGR modules)
│   └── loss.py             # Scale-and-Location Sensitive Loss (SLSIoULoss)
├── utils/
│   ├── __init__.py
│   ├── data.py             # PyTorch Dataset loader (IRSTD-1k, NUAA-SIRST, etc.)
│   └── metric.py           # Metrics computation (mIoU, nIoU, Pd, Fa)
├── train.py                # Main training script
├── test.py                 # Evaluation & inference script
├── requirements.txt        # Minimal Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Documentation
```

---

## Environment & Installation

### Requirements
- Python $\ge$ 3.8
- PyTorch $\ge$ 1.10.0
- torchvision $\ge$ 0.11.0
- `pytorch-wavelets` $\ge$ 1.3.0

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Dataset Preparation

Organize your IRSTD dataset (e.g., IRSTD-1k or NUAA-SIRST) under `./dataset/IRSTD-1k` as follows:

```text
dataset/IRSTD-1k/
├── trainval.txt
├── test.txt
├── images/
│   ├── 000001.png
│   └── ...
└── masks/
    ├── 000001.png
    └── ...
```

### 2. Training

To train SEG-UNet on your dataset:

```bash
python train.py --dataset-dir ./dataset/IRSTD-1k
```

### 3. Model Evaluation

To evaluate a trained checkpoint on the validation/test set:

```bash
python test.py --dataset-dir ./dataset/IRSTD-1k \
               --weight-path ./experiments/SEG_UNet-exp_.../exp01/model_best_iou.pth
```

---

## Citation

If you find SEG-UNet helpful in your research, please cite our paper:

```bibtex
@article{duan2026segunet,
  title={SEG-UNet: Spectral--Edge Gated U-Net for Infrared Small-Target Segmentation},
  author={Duan, Yulang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026}
}
```

---

## Acknowledgements

We thank the open-source community and creators of IRSTD benchmark datasets (NUAA-SIRST, IRSTD-1k) and baseline architectures (MSHNet, UIU-Net, DNANet).
