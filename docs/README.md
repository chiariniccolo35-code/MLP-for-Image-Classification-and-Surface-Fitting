# Documentation

## REPORT_LAB_2_IMAGING.pdf

The complete technical report, covering two independent chapters:

### Chapter 1 — MLP for Image Classification (pages 1–20)
A systematic, sequential hyperparameter study on Fashion-MNIST:
1. **1.1** Number of dense layers (5 / 10 / 15)
2. **1.2** Neurons per layer (32 / 64 / 128), comparing 5- vs. 10-layer architectures
3. **1.3** Activation function (ReLU / Sigmoid / Tanh)
4. **1.4** Batch size (32 / 128 / 256 / 512)
5. **1.5** Number of training epochs (50 / 100)
6. **1.6** Optimizer (SGD / ADAM)
7. **1.7** Final considerations — head-to-head comparison of the three best candidate models

### Chapter 2 — INR for 3D Surface Fitting (pages 21–26)
1. **2.1** Periodic activations for INR — introduces SIREN and the limitations of ReLU/Tanh for implicit representations
2. **2.2** Initialization and distribution of activations — derives, via variance calculations and the Central Limit Theorem, the weight-initialization scheme needed to preserve activation distributions across SIREN layers
3. **2.3** Experiment — SIREN-based reconstruction of the Stanford bunny from its point cloud via Signed-Distance-Function fitting, compared across training lengths and two initialization variants

## Reading Guide

| If you want to understand... | Read... |
|---|---|
| Why Sigmoid fails and Tanh is preferred | Report Section 1.3 |
| Why ADAM is rejected despite better raw accuracy | Report Section 1.6 |
| The final best MLP configuration | Report Section 1.7 |
| Why ReLU/Tanh MLPs fail for implicit representations | Report Section 2 (intro) |
| The SIREN weight-initialization derivation | Report Section 2.2 |
| The bunny point-cloud reconstruction results | Report Section 2.3 |

## Code ↔ Report Mapping

| Report Section | Code |
|---|---|
| Chapter 1 (1.1 – 1.7) | `notebooks/MLP_Image_Classification.ipynb` |
| Chapter 2 (2.1 – 2.3) | `notebooks/SIREN_3D_Surface_Fitting.ipynb` |

## Author

**Niccolò Chiari**, Master's degree in Mathematics (Applied Curriculum), University of Bologna, A.Y. 2024/2025.

Course: *B3093 — Mathematical and Machine Learning Methods in Imaging*
