# [MICCAI 2024] Longitudinal Segmentation of MS Lesions via Temporal Difference Weighting

## Overview

This repository provides the implementation of the **Temporal Difference Weighting (TDW) block**, introduced in the paper **"Longitudinal Segmentation of MS Lesions via Temporal Difference Weighting"**, accepted for the [*Longitudinal Disease Tracking and Modelling with Medical Images and Data*](https://ldtm-miccai.github.io/) workshop at MICCAI 2024.

### Read the paper: &nbsp; &nbsp;   [![arXiv](https://img.shields.io/badge/arXiv-2404.03010-B31B1B.svg)](https://arxiv.org/abs/2409.13416)

## Introduction

Longitudinal medical imaging plays a vital role in tracking disease progression, treatment response, and patient outcomes over time. However, traditional segmentation methods often process each time point independently, leading to inconsistencies and missed temporal patterns. To address this, we introduce **Temporal Difference Weighting (TDW)**, a novel approach that enhances segmentation by explicitly modeling temporal changes between consecutive imaging scans.

## Methodology

Our approach integrates a **Temporal Difference Weighting (TDW) block** into deep learning-based segmentation models, enabling them to leverage longitudinal information more effectively. TDW computes **normalized attention maps** from the differences between feature representations extracted at separate time points. This mechanism highlights areas of change between scans and enables improved segmentation performance.

The TDW block is designed to be modular and can be incorporated into various backbone architectures, such as the UNet. This repository provides implementations of:
- The **TDW block** as a standalone module.
- A **longitudinal UNet (LongiUNet)** without TDW.
- A **longitudinal UNet with TDW (LongiUNetDiffWeighting)** for enhanced feature merging.

## Usage

Clone and install the repository:

```bash
git clone https://github.com/MIC-DKFZ/Longitudinal-Difference-Weighting.git
cd Longitudinal-Difference-Weighting
pip install -e .
```

Import the block or the network:

```python
from difference_weighting.building_blocks.difference_weighting_block import DifferenceWeightingBlock
from difference_weighting.architectures.longi_unet import LongiUNet
from difference_weighting.architectures.longi_unet_difference_weighting import LongiUNetDiffWeighting
```

An implementation of the whole longitudinal segmentation framework will be made available after the next major nnUNet update.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{rokuss2024longitudinal,
  title={Longitudinal segmentation of MS lesions via temporal Difference Weighting},
  author={Rokuss, Maximilian and Kirchhoff, Yannick and Roy, Saikat and Kovacs, Balint and Ulrich, Constantin and Wald, Tassilo and Zenk, Maximilian and Denner, Stefan and Isensee, Fabian and Vollmuth, Philipp and Kleesiek, Jens and Maier-Hein, Klaus},
  journal={arXiv preprint arXiv:2409.13416},
  year={2024}
}
```

_Copyright © German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license:_
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)