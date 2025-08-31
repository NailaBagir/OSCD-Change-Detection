# OSCD Change Detection with Siamese U-Net

This project implements a deep learning pipeline for **bitemporal change detection**
using the [Onera Satellite Change Detection (OSCD) dataset](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection-dataset).

## Dataset
- Sentinel-2 satellite imagery
- 24 urban areas worldwide
- Two timestamps per city (before/after)
- Pixel-level binary change maps (building changes)

In this project:
- Training: all `train_labels` cities
- Validation: 50% of test cities
- Held-out Testing: remaining 50% of test cities

## Method
- **Model:** Siamese U-Net (two-branch encoder, shared weights, subtract at bottleneck, symmetric decoder)
- **Input:** 13 spectral bands (Sentinel-2)
- **Tile size:** 128×128 with stride 128
- **Losses:** BCE + Dice, Focal Loss
- **Metrics:** IoU, F1-score, Precision, Recall

## Results
- Precision and Recall can be improved by tuning threshold and loss functions.
- Visualizations show predicted change maps overlayed on RGB composites.

