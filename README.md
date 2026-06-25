# MLP for Image Classification and Surface Fitting

Two neural-network studies implemented in TensorFlow/Keras: (1) a systematic hyperparameter study of **Multilayer Perceptrons (MLPs)** for image classification on Fashion-MNIST, and (2) **Implicit Neural Representations (INR)** using **SIREN** (Sinusoidal Representation Networks) for 3D surface fitting from a point cloud.

## Project Overview

This project is split into two independent chapters, both centered on feedforward neural networks but addressing very different tasks:

1. **MLP for Image Classification (Chapter 1)** — a systematic, one-hyperparameter-at-a-time study (dense layers → neurons → activation function → batch size → epochs → optimizer) to find the best MLP configuration for classifying Fashion-MNIST images.
2. **INR for 3D Surface Fitting (Chapter 2)** — using a SIREN network (periodic/sine activations) to fit a Signed Distance Function (SDF) and reconstruct a 3D bunny shape from its point cloud, including a theoretical derivation of the SIREN weight-initialization scheme.

## Chapter 1 — MLP for Image Classification

**Dataset:** Fashion-MNIST — 60,000 training / 10,000 test grayscale 28×28 images, 10 clothing classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

**Evaluation criteria used throughout:**
- **Model complexity** — total number of parameters
- **Learning measures** — training vs. validation loss/accuracy (to detect over/underfitting)
- **Generalization measure** — test-set accuracy
- **Training speed**

The study proceeds hyperparameter-by-hyperparameter, each time fixing the previously-found best choice and exploring the next:

### 1.1 — Number of Dense Layers
Three models (5, 10, 15 dense layers; 32 neurons/layer, ReLU, batch size 32, SGD) are compared. The 5-layer model is selected as the best starting point — the 15-layer model is severely underfit, the 10-layer model shows mild overfitting.

### 1.2 — Number of Neurons per Layer
For both the 5- and 10-layer architectures, 32/64/128 neurons-per-layer are compared pairwise. The 5-dense-layer architecture is retained as the better generalizing choice across all neuron counts tested.

### 1.3 — Activation Function
ReLU, Sigmoid, and Tanh are compared on the 5-dense-layer models (32/64/128 neurons). Sigmoid fails to learn entirely (severely underperforming); Tanh consistently outperforms ReLU in validation/test accuracy across all three neuron counts.

### 1.4 — Batch Size
Batch sizes 32, 128, 256, 512 are compared (5-dense-layer, Tanh activation, all three neuron counts). Batch size 128 is selected as the best trade-off between accuracy and training-validation stability (smaller train/validation gap than batch size 32, while still reasonably accurate).

### 1.5 — Number of Training Epochs
50 vs. 100 epochs are compared (5-dense-layer, Tanh, batch size 128, all three neuron counts). 100 epochs consistently improves accuracy across the board, with only mild additional overfitting.

### 1.6 — Optimizer: SGD vs. ADAM
SGD and ADAM are compared on the final configuration (5-dense-layer, Tanh, batch size 128, 100 epochs, all three neuron counts). ADAM achieves much lower training loss and higher test accuracy, but at the cost of a much larger train/validation gap (i.e. overfitting); SGD is selected as the more balanced/robust optimizer.

### 1.7 — Final Considerations
The final three candidate models (5-dense-layer, 32/64/128 neurons, Tanh, batch size 128, 100 epochs, SGD) are compared head-to-head, and the **128-neuron model** is selected as the best overall — lowest training loss and best generalization to the test set among the three.

## Chapter 2 — INR for 3D Surface Fitting

### Background: Implicit Neural Representations

Unlike classical MLPs (used for classification/regression on discretely-represented signals), an **Implicit Neural Representation** parametrizes a signal as a *continuous function* Φ mapping coordinates to the quantity of interest (e.g. pixel coordinate → RGB color, or 3D coordinate → signed distance to a surface), implicitly defined by:

```
F(x, Φ, ∇ₓΦ, ∇²ₓΦ, ...) = 0,      Φ: x → Φ(x), x ∈ Rᵘ
```

Classical ReLU/Tanh-based architectures are shown to be poorly suited for this task — ReLU networks are piecewise-linear (their second derivative is zero everywhere), and Tanh, while capable of representing higher-order derivatives, behaves poorly in practice and fails to capture fine detail.

### 2.1 — Periodic Activations: SIREN

**SIREN** (Sinusoidal Representation Networks) uses the **sine** function as a periodic activation:

```
Φ(x) = Wₙ(φₙ₋₁ ∘ φₙ₋₂ ∘ ... ∘ φ₀)(x) + bₙ,      φₗ(xₗ) = sin(ω₀ Wₗxₗ + bₗ)
```

The frequency scaling factor **ω₀** accelerates training; the report identifies **ω₀ = 30** as the best value for the bunny-reconstruction experiment.

### 2.2 — Initialization and Distribution of Activations

The report derives, from first principles, the weight-initialization scheme needed to **preserve the distribution of activations across layers** (so the network's output statistics don't depend on depth):

- **First layer:** input `x ~ U(-1, 1)` i.i.d.; choosing weights `W ~ U(-c/fan-in, c/fan-in)` with `c = √3 · (fan-in/ω₀)` makes `ω₀Wx` approximately **standard normal** (via the Central Limit Theorem / weak Lindeberg condition), so that `sin(ω₀Wx)` is approximately **arcsine-distributed** on (−1, 1).
- **Subsequent layers:** with input now arcsine-distributed, a different constant `c = √(6/fan-in)/ω₀` is derived so that `W ~ U(-c, c)` again yields `ω₀Wx ~ N(0,1)` and hence `sin(ω₀Wx) ~ arcsin(-1,1)` — i.e. the distribution is preserved through depth.

This matches the SIREN paper's initialization scheme, derived here explicitly via variance calculations for uniform and arcsine-distributed random variables.

### 2.3 — Experiment: 3D Bunny Reconstruction

A **5-layer SIREN, 32 neurons/layer** (ω₀ = 30) is trained to fit a Signed Distance Function (SDF) representing the Stanford bunny, learned from its point cloud, and the surface is then extracted (e.g. via marching cubes) and visualized.

Two initialization schemes are compared:
- The derived initialization (first layer `W ~ U(-√(9/fan-in)/ω₀, √(9/fan-in)/ω₀)`, per Section 2.2)
- A **slightly different first-layer distribution**, `W ~ U(-√(3/fan-in)/ω₀, √(3/fan-in)/ω₀)`, which the report finds gives a **noticeably better bunny reconstruction**

Both variants are evaluated at **1,000 / 5,000 / 10,000 epochs**, showing progressively sharper surface reconstruction with more training.

## Project Structure

```
.
├── README.md                                  # This file
├── docs/
│   ├── REPORT_LAB_2_IMAGING.pdf               # Full technical report (Chapters 1-2)
│   └── README.md
├── notebooks/
│   ├── MLP_Image_Classification.ipynb         # Chapter 1: Fashion-MNIST MLP hyperparameter study
│   ├── SIREN_3D_Surface_Fitting.ipynb          # Chapter 2: SIREN / bunny point-cloud reconstruction
│   └── README.md
└── results/
    ├── REPORT_INR/                             # Figures for every hyperparameter configuration tested
    │   ├── First configuration/                # Dense layers (5/10/15)
    │   ├── Second configuration/                # Neurons per layer (32/64/128)
    │   ├── Third configuration/                 # Activation function (ReLU/Sigmoid/Tanh)
    │   ├── Fourth configuration/                 # Batch size (32/128/256/512)
    │   ├── Fifth configuration/                  # Epochs (50/100)
    │   └── Sixth configuration/                  # Optimizer (SGD/ADAM)
    └── README.md
```

> Note: despite the folder name `REPORT_INR`, these figures correspond to the **MLP/Fashion-MNIST hyperparameter study (Chapter 1)**, not the SIREN/INR experiment — the bunny-reconstruction figures are embedded directly in the report PDF rather than provided as separate image files.

## How to Use

### Prerequisites

- Python 3.x with TensorFlow / Keras
- For the 3D surface-fitting notebook: `pyvista`, `plyfile`, `scikit-image` (marching cubes), and a virtual display (`xvfb`) if running headless — the notebook installs these automatically when run on Google Colab

### Running the MLP Image Classification Notebook

```bash
jupyter notebook notebooks/MLP_Image_Classification.ipynb
```

Loads Fashion-MNIST directly via `tensorflow.keras.datasets`, then builds/trains/evaluates the sequence of MLP configurations described in Chapter 1.

### Running the SIREN 3D Surface Fitting Notebook

```bash
jupyter notebook notebooks/SIREN_3D_Surface_Fitting.ipynb
```

Loads a 3D point cloud (bunny), builds the SIREN network, trains it to fit the corresponding Signed Distance Function, and extracts/visualizes the reconstructed surface.

## Author

**Niccolò Chiari**  
Master's degree in Mathematics (Applied Curriculum)  
University of Bologna, Academic Year 2024/2025

Course: *B3093 — Mathematical and Machine Learning Methods in Imaging*

## References

- Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., & Wetzstein, G. (2020). "Implicit Neural Representations with Periodic Activation Functions" (SIREN). *NeurIPS*.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms." *arXiv:1708.07747*.

## License

Educational project. Available for academic use.

---

For complete quantitative results (accuracy/loss tables for every configuration), see `results/README.md`. For full derivations and all figures, consult `docs/REPORT_LAB_2_IMAGING.pdf`.
