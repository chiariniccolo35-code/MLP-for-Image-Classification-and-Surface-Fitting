# Results — MLP Hyperparameter Study (Fashion-MNIST)

This folder contains the figures (loss/accuracy curves) for every configuration tested in **Chapter 1** of the report. The folder is named `REPORT_INR` in the original project, but its contents are entirely the **MLP/Fashion-MNIST classification** results, not the SIREN/INR experiment.

Metrics abbreviations used throughout: **trL/trA** = training loss/accuracy, **vL/vA** = validation loss/accuracy, **teL/teA** = test loss/accuracy.

## First configuration — Number of Dense Layers

`First configuration/5dl_32n.png`, `10dl_32n.png`, `15dl_32n.png`

Base setup: 32 neurons/layer, ReLU, batch size 32, SGD.

| Layers | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| 5 | 32.71% | 88.22% | 36.47% | 86.73% | 40.18% | 85.32% |
| 10 | 33.78% | 87.98% | 40.14% | 86.18% | 44.09% | 85.12% |
| 15 | 72.27% | 74.66% | 78.03% | 73.25% | 82.01% | 72.17% |

**Parameter counts:** 5-layer = 29,674 · 10-layer = 34,954 · 15-layer = 40,234

**Conclusion:** 5-layer model performs best overall; 10-layer shows mild overfitting; 15-layer is severely underfit.

## Second configuration — Neurons per Layer

`Second configuration/{5,10}dl_{32,64,128}n.png`

Comparing 5-layer vs. 10-layer models at each neuron count (ReLU, batch size 32, SGD).

**32 neurons:**

| Model | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| 5-layer | 32.71% | 88.22% | 36.47% | 86.73% | 40.18% | 85.32% |
| 10-layer | 33.78% | 87.98% | 40.14% | 86.18% | 44.09% | 85.12% |

→ 5-layer better (10-layer shows mild overfitting).

**64 neurons:**

| Model | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| 5-layer | 30.02% | 89.08% | 35.14% | 87.3% | 38.77% | 86.12% |
| 10-layer | 28.12% | 89.77% | 36.04% | 87.67% | 39.1% | 86.41% |

→ 10-layer slightly better numerically but with mild overfitting; 5-layer chosen as the more balanced option.

**128 neurons:**

| Model | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| 5-layer | 28.97% | 89.37% | 35.3% | 87.4% | 39.71% | 85.87% |
| 10-layer | 26.6% | 90.2% | 36.6% | 87.3% | 40.9% | 86.27% |

→ 10-layer looks stronger on paper, but 5-layer generalizes better to new data — 5-layer retained as the preferred architecture going forward.

## Third configuration — Activation Function

`Third configuration/{32n,64n,128n}/{ReLU,Sigmoid,Tanh}.png`

All on the 5-dense-layer architecture, batch size 32, SGD.

**32 neurons:**

| Activation | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| ReLU | 32.71% | 88.22% | 36.47% | 86.73% | 40.2% | 85.33% |
| Sigmoid | 230.14% | 14.8% | 230.14% | 14.33% | 230.14% | 14.48% |
| Tanh | 31.66% | 88.89% | 37.39% | 86.66% | 40.37% | 85.54% |

→ Sigmoid fails completely (no learning). Tanh is the best choice; Sigmoid excluded from further testing.

**64 neurons:**

| Activation | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| ReLU | 30.02% | 89.08% | 35.14% | 87.3% | 38.77% | 86.12% |
| Tanh | 28.33% | 90.14% | 34.08% | 87.88% | 38.17% | 86.55% |

→ Tanh better on every metric.

**128 neurons:**

| Activation | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| ReLU | 28.97% | 89.37% | 35.3% | 87.4% | 39.71% | 85.87% |
| Tanh | 25.2% | 91.1% | 32.6% | 88.42% | 36.24% | 87.06% |

→ Tanh again better across the board. **Tanh selected as the best activation function for all subsequent experiments.**

## Fourth configuration — Batch Size

`Fourth configuration/{32,64,128}neurons/{32,128,256,512}bs.png`

All on 5-dense-layer, Tanh activation.

**32 neurons:**

| Batch Size | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| 32 | 31.66% | 88.89% | 37.39% | 86.66% | 40.37% | 85.54% |
| 128 | 48.3% | 83.68% | 49.09% | 83.13% | 52.22% | 82% |
| 256 | 60.47% | 80.29% | 60.25% | 80.67% | 62.57% | 79.5% |
| 512 | 79.15% | 75.25% | 78.86% | 75.50% | 80.77% | 74.58% |

→ Batch size 32 gives the highest raw accuracy, but batch size 128 gives the best train/validation **stability** (smaller gap) — selected as the best trade-off.

**64 neurons:**

| Batch Size | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| 32 | 28.33% | 90.14% | 34.08% | 87.88% | 38.17% | 86.55% |
| 128 | 41.89% | 85.29% | 42.84% | 84.77% | 46.48% | 83.60% |
| 256 | 53.64% | 81.71% | 53.22% | 81.65% | 56.38% | 80.27% |
| 512 | 64.26% | 79.20% | 67.87% | 79.59% | 66.48% | 78.04% |

→ Batch size 128 again selected as the best choice.

**128 neurons:**

| Batch Size | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| 32 | 25.2% | 91.09% | 32.56% | 88.42% | 36.24% | 87.06% |
| 128 | 39% | 86.31% | 39.55% | 85.77% | 44% | 84.1% |
| 256 | 46.96% | 83.39% | 47.17% | 83.25% | 50.17% | 82% |
| 512 | 57.66% | 79.62% | 57.19% | 79.77% | 59.7% | 78.54% |

→ Same pattern as the other two architectures; batch size 128 retained.

## Fifth configuration — Number of Epochs

`Fifth configuration/{32,64,128} neurons/{50,100}epochs.png`

All on 5-dense-layer, Tanh, batch size 128.

**32 neurons:**

| Epochs | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| 50 | 48.3% | 83.68% | 49.09% | 83.13% | 52.22% | 81.99% |
| 100 | 38.18% | 86.78% | 39.37% | 86.23% | 43.8% | 84.94% |

**64 neurons:**

| Epochs | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| 50 | 41.89% | 85.29% | 42.83% | 84.77% | 46.48% | 83.6% |
| 100 | 34.94% | 87.91% | 37.49% | 86.79% | 41.18% | 85.1% |

**128 neurons:**

| Epochs | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| 50 | 39% | 86.31% | 39.55% | 85.77% | 44% | 84.1% |
| 100 | 32.44% | 88.49% | 35.64% | 87.33% | 39.49% | 85.77% |

→ 100 epochs consistently improves every metric, with only mild additional overfitting — **100 epochs retained** for all subsequent experiments.

## Sixth configuration — Optimizer (SGD vs. ADAM)

`Sixth configuration/{32,64,128} neurons/{SGD,ADAM}.png`

All on 5-dense-layer, Tanh, batch size 128, 100 epochs.

**32 neurons:**

| Optimizer | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| SGD | 38.18% | 86.78% | 39.37% | 86.23% | 43.8% | 84.94% |
| ADAM | 13.4% | 95.23% | 55.18% | 86.99% | 60.14% | 85.97% |

**64 neurons:**

| Optimizer | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| SGD | 34.94% | 87.91% | 37.49% | 86.79% | 41.18% | 85.1% |
| ADAM | 6.4% | 97.55% | 57.96% | 88.34% | 64.42% | 87.16% |

**128 neurons:**

| Optimizer | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|
| SGD | 32.44% | 88.49% | 35.64% | 87.33% | 39.49% | 85.77% |
| ADAM | 5.56% | 97.91% | 54.18% | 88.57% | 59.2% | 87.66% |

→ ADAM achieves much lower training loss and (slightly) higher test accuracy, but the very large train/validation gap signals **strong overfitting**. **SGD is selected as the more robust/balanced optimizer.**

## Final Model Comparison (Section 1.7)

All three final candidates: 5-dense-layer, Tanh, batch size 128, 100 epochs, SGD:

| Model | Neurons | trL | trA | vL | vA | teL | teA |
|---|---|---|---|---|---|---|---|
| Model 1 | 32 | 38.18% | 86.78% | 39.37% | 86.23% | 43.8% | 84.94% |
| Model 2 | 64 | 34.94% | 87.91% | 37.49% | 86.79% | 41.18% | 85.1% |
| **Model 3** | **128** | **32.44%** | **88.49%** | **35.64%** | **87.33%** | **39.49%** | **85.77%** |

**Best overall model: Model 3** — 5 dense layers, 128 neurons/layer, Tanh activation, batch size 128, 100 epochs, SGD optimizer. Lowest training loss and best generalization to the test set among the three.

## Notes on the SIREN / 3D Surface-Fitting Results (Chapter 2)

The bunny-reconstruction figures (1,000 / 5,000 / 10,000 epochs, for both the original and the modified first-layer initialization) are embedded directly in `docs/REPORT_LAB_2_IMAGING.pdf` (Section 2.3) and are not provided as separate image files in this results folder.

## References

For the full discussion behind every table above, see:
- Main `README.md` — project overview
- `docs/REPORT_LAB_2_IMAGING.pdf` — Chapter 1 (full discussion, all figures referenced by table)
- `notebooks/MLP_Image_Classification.ipynb` — code that generated these results
